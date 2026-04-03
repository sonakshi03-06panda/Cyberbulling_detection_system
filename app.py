import os
import json
import re
import unicodedata
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, session
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from werkzeug.middleware.proxy_fix import ProxyFix
from googleapiclient.discovery import build
import torch
import pandas as pd
from dotenv import load_dotenv
import io

import emoji
import ftfy
import contractions

import database

# ── Load environment ─────────────────────────────────────────────────────────
load_dotenv()

import tempfile
secret_json_content = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRETS_JSON")
if secret_json_content:
    tmp_path = os.path.join(tempfile.gettempdir(), "client_secrets.json")
    with open(tmp_path, "w") as f:
        f.write(secret_json_content)
    os.environ["GOOGLE_OAUTH_CLIENT_SECRETS_FILE"] = tmp_path

# Allow OAuth over HTTP for local development
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-please-change")

# Hugging Face Spaces proxies requests and embeds the app inside an iframe.
# 1) Use ProxyFix to correctly determine client IP, Host, and Proto so url_for works correctly
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# 2) Allow session cookies across domains to prevent session dropping during OAuth
app.config.update(
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True
)

@app.context_processor
def inject_auth():
    return dict(is_authenticated='credentials' in session)

# ── Model config ─────────────────────────────────────────────────────────────
MODEL_PATH = "NiketGupta06/VibeCheck"

print("Loading ToxicBERT model from Hugging Face Hub...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded successfully!", flush=True)

# ── Database Init ────────────────────────────────────────────────────────────
database.init_db()
print("Database initialized.", flush=True)

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ── Preprocessing pipeline (matches training notebook CyberB_multi.ipynb) ────
LEET_MAP = {
    'a': '[4@\u00e0\u00e1\u00e2\u00e3\u00e4\u00e5]', 'b': '[8]',      'e': '[3\u20ac\u00e8\u00e9\u00ea\u00eb]',
    'i': '[1!|\u00ec\u00ed\u00ee\u00ef]',             'o': '[0\u00f2\u00f3\u00f4\u00f5\u00f6]', 's': '[5$]',
    't': '[7+]',                                       'g': '[69]',      'z': '[2]',
}


def _normalize_unicode(text: str) -> str:
    text = ftfy.fix_text(text)
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    return text


def _replace_urls(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', '<URL>', text)


def _replace_mentions(text: str) -> str:
    return re.sub(r'@[\w_]+', '<USER>', text)


def _split_hashtags(text: str) -> str:
    def split_tag(match):
        tag = match.group(1)
        tag = re.sub(r'([A-Z])', r' \1', tag)
        return tag.lower()
    return re.sub(r'#(\w+)', split_tag, text)


def _expand_contractions(text: str) -> str:
    try:
        return contractions.fix(text)
    except Exception:
        return text


def _expand_emojis(text: str) -> str:
    text = emoji.demojize(text, language='en')
    text = re.sub(r':([a-zA-Z0-9_]+):', r' \1 ', text)
    text = text.replace('_', ' ')
    return text


def _decode_leetspeak(text: str) -> str:
    for char, pattern in LEET_MAP.items():
        text = re.sub(pattern, char, text)
    return text


def _collapse_separators(text: str) -> str:
    return re.sub(r'(\w)[\.\-_/\\]+(\w)', r'\1\2', text)


def _fix_spaced_words(text: str) -> str:
    return re.sub(r'\b(\w\s){2,}\w\b', lambda m: m.group(0).replace(' ', ''), text)


def _collapse_repetitions(text: str) -> str:
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'([!?.,]){2,}', r'\1', text)
    return text


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline — matches training conditions in CyberB_multi.ipynb."""
    text = str(text)
    text = _normalize_unicode(text)
    text = _replace_urls(text)
    text = _replace_mentions(text)
    text = _split_hashtags(text)
    text = text.lower()
    text = _expand_contractions(text)
    text = _expand_emojis(text)
    text = _decode_leetspeak(text)
    text = _collapse_separators(text)
    text = _fix_spaced_words(text)
    text = _collapse_repetitions(text)
    text = re.sub(r'[^\w\s<>]', ' ', text)   # keep <URL> <USER> tokens
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ── YouTube API ───────────────────────────────────────────────────────────────
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")


def get_video_id(url: str):
    """Extract video ID from a YouTube URL."""
    if not url:
        return None
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    if "youtu.be/" in url:
        vid = url.split("youtu.be/")[1]
        return vid.split("?")[0]
    return None


def get_video_title(video_id: str) -> str:
    """Fetch the video title from YouTube API."""
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        response = youtube.videos().list(part="snippet", id=video_id).execute()
        items = response.get("items", [])
        if items:
            return items[0]["snippet"]["title"]
    except Exception:
        pass
    return "Unknown Title"


def get_user_videos(credentials_dict: dict) -> list:
    """Fetch user's uploaded videos."""
    try:
        credentials = Credentials(
            token=credentials_dict.get("token"),
            refresh_token=credentials_dict.get("refresh_token"),
            token_uri=credentials_dict.get("token_uri"),
            client_id=credentials_dict.get("client_id"),
            client_secret=credentials_dict.get("client_secret"),
            scopes=credentials_dict.get("scopes")
        )
        youtube = build("youtube", "v3", credentials=credentials)
        channels_response = youtube.channels().list(part="contentDetails", mine=True).execute()
        items = channels_response.get("items", [])
        if not items:
            return []
        uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        
        playlist_response = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist_id,
            maxResults=50
        ).execute()
        
        videos = []
        for item in playlist_response.get("items", []):
            snippet = item["snippet"]
            videos.append({
                "video_id": snippet["resourceId"]["videoId"],
                "title": snippet["title"],
                "thumbnail_url": snippet["thumbnails"].get("medium", {}).get("url", "")
            })
        return videos
    except Exception as e:
        print(f"Error fetching user videos: {e}")
        return []


def get_comments(video_id: str, max_comments: int = 2000) -> list:
    """Fetch up to max_comments top-level comments for a YouTube video."""
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    next_page = None

    while len(comments) < max_comments:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page,
            textFormat="plainText",
        )
        response = req.execute()

        for item in response.get("items", []):
            t_comment = item["snippet"]["topLevelComment"]
            comments.append({
                "id": t_comment["id"],
                "text": t_comment["snippet"]["textDisplay"]
            })
            if len(comments) >= max_comments:
                break

        next_page = response.get("nextPageToken")
        if not next_page:
            break

    return comments


def predict(comment: str) -> dict:
    """Run ToxicBERT inference on a single comment. Returns dict of 6 label scores.

    Applies the same preprocessing pipeline used during model training
    (CyberB_multi.ipynb) before tokenization.
    """
    clean = preprocess_text(comment)
    inputs = tokenizer(
        clean,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0]
    return {label: float(probs[i]) for i, label in enumerate(LABELS)}


# ── In-memory dashboard cache (per analysis session) ─────────────────────────
_dashboard_cache: dict = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "GET":
        videos = []
        if "credentials" in session:
            videos = get_user_videos(session["credentials"])
        return render_template("analyze.html", videos=videos)

    # POST — run analysis
    url = request.form.get("url", "").strip()
    video_id_post = request.form.get("video_id", "").strip()

    if video_id_post:
        url = f"https://www.youtube.com/watch?v={video_id_post}"

    if not url:
        flash("Please enter a YouTube URL.", "error")
        return render_template("analyze.html", error="Please enter a YouTube URL.")

    video_id = get_video_id(url)
    if not video_id:
        flash("Invalid YouTube URL. Please enter a valid link.", "error")
        return render_template("analyze.html", error="Invalid YouTube URL.")

    if not YOUTUBE_API_KEY:
        flash("YouTube API key not configured.", "error")
        return render_template("analyze.html", error="YouTube API key not configured.")

    try:
        # Fetch comments
        fetched_data = get_comments(video_id, max_comments=2000)
        if not fetched_data:
            flash("No comments found or comments are disabled for this video.", "error")
            return render_template("analyze.html", error="No comments found.")

        # Run inference
        results = []
        comments_texts = [c["text"] for c in fetched_data]
        comments_ids = [c["id"] for c in fetched_data]

        for c in comments_texts:
            p = predict(c)
            results.append(p)

        df = pd.DataFrame(results)
        df["comment"] = comments_texts
        df["comment_id"] = comments_ids

        # Overall toxicity score (sum of all label scores)
        df["toxicity_score"] = df[LABELS].sum(axis=1)

        # Statistics
        total_comments = len(df)
        toxic_percent = round(float((df["toxic"] > 0.5).mean() * 100), 1)
        category_means = {k: round(float(v), 4) for k, v in df[LABELS].mean().to_dict().items()}

        top_toxic = (
            df.sort_values("toxicity_score", ascending=False)
            .head(5)[["comment", "toxicity_score"]]
            .to_dict(orient="records")
        )
        top_threats = (
            df.sort_values("threat", ascending=False)
            .head(5)[["comment", "threat"]]
            .to_dict(orient="records")
        )
        comment_rows = (
            df.sort_values("toxicity_score", ascending=False)
            .head(50)[["comment_id", "comment", "toxicity_score"] + LABELS]
            .to_dict(orient="records")
        )

        # Fetch video title
        video_title = get_video_title(video_id)

        # Save to DB
        analysis_id = database.save_analysis(
            video_url=url,
            video_id=video_id,
            video_title=video_title,
            total_comments=total_comments,
            toxic_percent=toxic_percent,
            results_json=json.dumps(comment_rows),
        )

        # Cache for this session
        dashboard_data = {
            "analysis_id": analysis_id,
            "video_url": url,
            "video_id": video_id,
            "video_title": video_title,
            "total_comments": total_comments,
            "toxic_percent": toxic_percent,
            "chart_data": category_means,
            "top_category": _get_top_category(category_means),
            "top_comments": top_toxic,
            "top_threats": top_threats,
            "comment_rows": comment_rows,
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        _dashboard_cache["latest"] = dashboard_data

        return redirect(url_for("dashboard"))

    except Exception as e:
        error_msg = str(e)
        if "quotaExceeded" in error_msg or "forbidden" in error_msg.lower():
            msg = "YouTube API quota exceeded. Please try again tomorrow."
        elif "videoNotFound" in error_msg or "commentsDisabled" in error_msg:
            msg = "This video is private, not found, or has comments disabled."
        else:
            msg = f"Analysis failed: {error_msg}"
        flash(msg, "error")
        return render_template("analyze.html", error=msg)


@app.route("/dashboard", methods=["GET"])
def dashboard():
    analysis_id = request.args.get("id")

    if analysis_id:
        # Load from DB (history view)
        row = database.get_analysis_by_id(int(analysis_id))
        if not row:
            flash("Analysis not found.", "error")
            return redirect(url_for("history"))
        comment_rows = json.loads(row["results_json"]) if row["results_json"] else []
        computed_chart = _compute_chart_data(comment_rows)
        data = {
            "analysis_id": row["id"],
            "video_url": row["video_url"],
            "video_id": row["video_id"],
            "video_title": row["video_title"] or "Unknown",
            "total_comments": row["total_comments"],
            "toxic_percent": row["toxic_percent"],
            "chart_data": computed_chart,
            "top_category": _get_top_category(computed_chart),
            "top_comments": sorted(comment_rows, key=lambda x: x.get("toxicity_score", 0), reverse=True)[:5],
            "top_threats": sorted(comment_rows, key=lambda x: x.get("threat", 0), reverse=True)[:5],
            "comment_rows": comment_rows[:50],
            "analyzed_at": row["analyzed_at"],
        }
    else:
        # Load from session cache
        data = _dashboard_cache.get("latest")
        if not data:
            flash("No analysis found. Please analyze a video first.", "error")
            return redirect(url_for("analyze"))
        data["top_category"] = _get_top_category(data.get("chart_data", {}))

    return render_template("dashboard.html", **data)


def _compute_chart_data(comment_rows: list) -> dict:
    """Recompute category means from stored comment rows."""
    if not comment_rows:
        return {label: 0 for label in LABELS}
    df = pd.DataFrame(comment_rows)
    means = {}
    for label in LABELS:
        if label in df.columns:
            means[label] = round(float(df[label].mean()), 4)
        else:
            means[label] = 0
    return means


def _get_top_category(chart_data: dict) -> str:
    """Return the label name with the highest average score."""
    if not chart_data:
        return "—"
    top = max(chart_data, key=chart_data.get)
    return top.replace("_", " ").title()


@app.route("/history", methods=["GET"])
def history():
    analyses = database.get_all_analyses()
    return render_template("history.html", analyses=analyses)


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")


# ── OAuth 2.0 ────────────────────────────────────────────────────────────────

@app.route("/login")
def login():
    client_secrets_file = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRETS_FILE", "client_secrets.json")
    if not os.path.exists(client_secrets_file):
        flash("YouTube login is not configured on this server (missing client_secrets.json).", "error")
        return redirect(url_for("home"))
    
    flow = Flow.from_client_secrets_file(
        client_secrets_file,
        scopes=[
            "https://www.googleapis.com/auth/youtube.force-ssl",
            "https://www.googleapis.com/auth/youtube.readonly"
        ]
    )
    # Generate dynamic redirect URI and enforce HTTPS for production
    redirect_uri = url_for('oauth2callback', _external=True)
    if "localhost" not in redirect_uri and redirect_uri.startswith("http://"):
        redirect_uri = redirect_uri.replace("http://", "https://", 1)
    flow.redirect_uri = redirect_uri
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true"
    )
    session["state"] = state
    if hasattr(flow, "code_verifier"):
        session["code_verifier"] = flow.code_verifier
        
    return redirect(authorization_url)


@app.route("/oauth2callback")
def oauth2callback():
    client_secrets_file = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRETS_FILE", "client_secrets.json")
    state = session.get("state")
    if not state:
        return redirect(url_for("home"))

    flow = Flow.from_client_secrets_file(
        client_secrets_file,
        scopes=[
            "https://www.googleapis.com/auth/youtube.force-ssl",
            "https://www.googleapis.com/auth/youtube.readonly"
        ],
        state=state
    )
    # Generate dynamic redirect URI and enforce HTTPS for production
    redirect_uri = url_for('oauth2callback', _external=True)
    if "localhost" not in redirect_uri and redirect_uri.startswith("http://"):
        redirect_uri = redirect_uri.replace("http://", "https://", 1)
    flow.redirect_uri = redirect_uri
    
    # Restore the code_verifier if it was saved in the session (PKCE)
    if session.get("code_verifier"):
        flow.code_verifier = session["code_verifier"]

    authorization_response = request.url

    # OAUTHLIB_INSECURE_TRANSPORT is set at app startup for local dev

    try:
        flow.fetch_token(authorization_response=authorization_response)
        credentials = flow.credentials
        session["credentials"] = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes
        }
        flash("Successfully connected to YouTube!", "success")
    except Exception as e:
        flash(f"OAuth failed: {e}", "error")

    return redirect(url_for("home"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("home"))


@app.route("/api/moderate", methods=["POST"])
def moderate():
    if not session.get("credentials"):
        return jsonify({"error": "Unauthorized. Please sign in with YouTube."}), 401

    data = request.get_json()
    if not data or "comment_id" not in data or "action" not in data:
        return jsonify({"error": "Invalid request"}), 400

    comment_id = data["comment_id"]
    action = data["action"]

    creds_data = session["credentials"]
    credentials = Credentials(
        token=creds_data.get("token"),
        refresh_token=creds_data.get("refresh_token"),
        token_uri=creds_data.get("token_uri"),
        client_id=creds_data.get("client_id"),
        client_secret=creds_data.get("client_secret"),
        scopes=creds_data.get("scopes")
    )

    try:
        youtube = build("youtube", "v3", credentials=credentials)
        if action == "delete" or action == "reject":
            # Video owners can only reject/hide comments, not truly delete them.
            # comments.delete() only works for comments YOU authored.
            youtube.comments().setModerationStatus(id=comment_id, moderationStatus="rejected").execute()
        else:
            return jsonify({"error": "Invalid moderation action"}), 400

        return jsonify({"success": True})
    except Exception as e:
        error_msg = str(e)
        if "forbidden" in error_msg.lower() or "403" in error_msg:
            return jsonify({"error": "Forbidden. You can only moderate comments on your own videos."}), 403
        return jsonify({"error": f"YouTube API Error: {error_msg}"}), 500


# ── Startup ───────────────────────────────────────────────────────────────────

database.init_db()

if __name__ == "__main__":
    app.run(debug=True)
