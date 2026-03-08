"""
Flask web app for toxicity reporting.
Run with: python app.py
Then visit http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify, send_file
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# load environment variables from .env (if present)
# Look for .env in parent directory (project root)
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
print(f"[DEBUG] Loading .env from: {env_path}")
print(f"[DEBUG] .env file exists: {os.path.exists(env_path)}")
load_dotenv(env_path)

from comment_fetcher import CommentFetcher
from toxicity_analyzer import ToxicityAnalyzer, analyze_comments
from report_generator import ToxicityReportGenerator

# Import conversation context modules (optional)
try:
    from context_integration import integrate_context_with_predictor
    CONTEXT_AVAILABLE = True
    print("[INFO] Conversation context module loaded successfully")
except ImportError:
    CONTEXT_AVAILABLE = False
    print("[WARNING] Conversation context module not available - will use standard analysis")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max


# Initialize analyzers (lazy loading)
analyzer = None
fetcher = None
context_pipeline = None
report_gen = ToxicityReportGenerator("reports")
_models_loaded = False  # Flag to track if models are loaded


def _load_models():
    """Lazy load models only on first use (not at startup)."""
    global analyzer, context_pipeline, _models_loaded
    
    if _models_loaded:
        return  # Already loaded
    
    print("[INFO] Loading ML models (this takes ~5-10 seconds)...")
    import time
    start = time.time()
    
    # Use final model (quantized version not available)
    analyzer = ToxicityAnalyzer("models/final_model")
    
    elapsed = time.time() - start
    print(f"[INFO] Models loaded in {elapsed:.1f}s")
    
    # Initialize context pipeline if available
    if CONTEXT_AVAILABLE:
        try:
            context_pipeline = integrate_context_with_predictor(
                analyzer,
                context_window_size=3,
                enable_analytics=False
            )
            print("[INFO] Context pipeline ready")
        except Exception as e:
            print(f"[WARNING] Context pipeline failed: {e}")
    
    _models_loaded = True


def init_services():
    """Initialize services (light setup, models load lazily)."""
    global fetcher
    
    # Initialize fetcher with API key from environment
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    if youtube_key:
        print(f"[INFO] YOUTUBE_API_KEY configured: {youtube_key[:20]}...")
    else:
        print("[WARNING] YOUTUBE_API_KEY not set. Comment fetching will fail.")
        print("[WARNING] Make sure YOUTUBE_API_KEY is in your .env file")
    
    fetcher = CommentFetcher(youtube_key)
    print("[INFO] Comment fetcher ready")
    print("[INFO] ✓ App startup complete! Models will load on first analysis request.")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vibe Check - YouTube Safety Monitor</title>
    <!-- Chart.js for report visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Times New Roman', Times, serif;
            background: #000000;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo-section h1 {
            color: #333;
            font-size: 28px;
            margin: 0;
            font-weight: 700;
        }
        
        .logo-section p {
            color: #888;
            font-size: 13px;
            margin: 5px 0 0 0;
        }
        
        nav {
            display: flex;
            gap: 30px;
        }
        
        nav a {
            color: #666;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            transition: color 0.3s;
            cursor: pointer;
        }
        
        nav a:hover {
            color: #667eea;
        }
        
        .main-container {
            flex: 1;
            max-width: 1200px;
            margin: 40px auto;
            padding: 0 20px;
            width: 100%;
        }
        
        .content-wrapper {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        .section {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .section h2 {
            color: #333;
            font-size: 22px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }
        input[type="url"], select, textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            font-family: inherit;
        }
        input[type="url"]:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .help-text {
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 25px;
        }
        button {
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #ff8a00 0%, #e52e71 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(255,138,0,0.4);
            transition: all 0.3s;
        }
        .btn-primary:hover {
            transform: translateY(-3px) scale(1.03);
            box-shadow: 0 8px 25px rgba(229,46,113,0.6);
        }
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #667eea;
        }
        .spinner {
            border: 4px solid #f0f0f0;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .alert {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
            animation: fadeIn 0.5s;
        }
        .alert-success {
            background: #e0f7fa;
            color: #00695c;
            border: 1px solid #4dd0e1;
        }
        .alert-error {
            background: #ffebee;
            color: #c62828;
            border: 1px solid #ef5350;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .results {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            display: none;
        }
        .report-link {
            display: inline-block;
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            text-decoration: none;
            margin-top: 15px;
            transition: all 0.3s;
        }
        .report-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            margin-right: 8px;
            font-weight: 500;
        }
        .badge-info {
            background: #e3f2fd;
            color: #1565c0;
        }
        .badge-warning {
            background: #fff3e0;
            color: #e65100;
        }
        .badge-danger {
            background: #ffebee;
            color: #c62828;
        }
        
        .reports-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 300px;
            text-align: center;
            color: #999;
        }
        
        .reports-section.active {
            align-items: flex-start;
            justify-content: flex-start;
        }
        
        .reports-icon {
            font-size: 48px;
            margin-bottom: 15px;
            opacity: 0.3;
        }
        
        footer {
            text-align: center;
            color: rgba(255, 255, 255, 0.6);
            padding: 30px 20px;
            font-size: 13px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        @media (max-width: 768px) {
            .content-wrapper {
                grid-template-columns: 1fr;
            }
            .header-content {
                flex-direction: column;
                gap: 15px;
            }
            nav {
                justify-content: center;
                gap: 15px;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <div class="logo-section">
                <h1>Vibe Check</h1>
                <p>Safety Monitor for GenZ</p>
            </div>
        </div>
    </header>
    
    <div class="main-container">
        <div class="content-wrapper">
            <!-- Left Column: Analyze Comments -->
            <div class="section">
                <h2>Analyze URL</h2>
                
                <div class="alert alert-error" id="errorAlert"></div>
                <div class="alert alert-success" id="successAlert"></div>
                
                <form id="reportForm">
                    <div class="form-group">
                        <label for="url">Enter a video URL to analyze comments for cyberbullying and toxic content</label>
                        <input 
                            type="url" 
                            id="url" 
                            name="url" 
                            placeholder="https://www.youtube.com/watch?v=..." 
                            required
                        >
                    </div>
                    
                    <div class="form-group">
                        <label for="maxComments">Max Comments to Analyze (0 = All)</label>
                        <input 
                            type="number" 
                            id="maxComments" 
                            name="maxComments" 
                            value="0" 
                            min="0"
                        >
                        <small style="color: #666; display: block; margin-top: 5px;">Enter 0 to analyze all comments, or specify a number limit</small>
                    </div>
                    
                    <div class="form-group" style="display: flex; align-items: center; gap: 10px;">
                        <input 
                            type="checkbox" 
                            id="useContext" 
                            name="useContext"
                            style="width: auto; cursor: pointer;"
                        >
                        <label for="useContext" style="margin: 0; cursor: pointer; flex: 1;">
                            Analyze with Conversation Context (More Accurate)
                        </label>
                    </div>
                    <small style="color: #999; display: block; margin-top: 5px;">Consider previous comments in thread for more accurate toxicity detection</small>
                    
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p>Analyzing comments...</p>
                    </div>
                    
                    <button type="submit" class="btn-primary" id="analyzeBtn">Analyze</button>
                </form>
            </div>
            
            <!-- Right Column: View Reports -->
            <div class="section">
                <h2>View Reports</h2>
                
                <div class="reports-section" id="reportsSection">
                    <div class="reports-icon">📊</div>
                    <p>Analysis results will appear here</p>
                </div>
                
                <div class="results" id="results">
                    <p id="summary" style="margin: 15px 0; color: #333;"></p>
                </div>
            </div>
        </div>
    </div>
    
    <footer>
        © 2026. Built with ❤️ by NSync
    </footer>
    
    <script>
        document.getElementById('reportForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const url = document.getElementById('url').value;
            const maxComments = document.getElementById('maxComments').value;
            const useContext = document.getElementById('useContext').checked;
            const analyzeBtn = document.getElementById('analyzeBtn');
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('reportsSection').style.display = 'none';
            document.getElementById('errorAlert').style.display = 'none';
            analyzeBtn.disabled = true;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url, maxComments, useContext })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Add context indicator if used
                    let contextIndicator = '';
                    if (data.use_context) {
                        contextIndicator = '<span class="badge" style="background: #c8e6c9; color: #2e7d32;">✓ Context-Aware Analysis</span>';
                    }
                    
                    document.getElementById('summary').innerHTML = `
                        <strong>Analysis Complete!</strong><br><br>
                        <span class="badge badge-info">${data.total_comments} Comments</span>
                        <span class="badge badge-danger">${data.toxic_comments} Toxic (${data.toxic_rate.toFixed(1)}%)</span>
                        <span class="badge badge-warning">Confidence: ${(data.avg_confidence * 100).toFixed(1)}%</span>
                        ${contextIndicator}
                        <hr style="margin: 20px 0; border: 1px solid #ddd;">
                    `;
                    
                    // Extract body content from the report HTML
                    const parser = new DOMParser();
                    const reportDoc = parser.parseFromString(data.report_html, 'text/html');
                    
                    // Extract styles from report
                    const styles = reportDoc.querySelectorAll('style');
                    styles.forEach(style => {
                        const newStyle = document.createElement('style');
                        newStyle.textContent = style.textContent;
                        document.head.appendChild(newStyle);
                    });
                    
                    // Extract body content
                    const bodyContent = reportDoc.body.innerHTML;
                    
                    // Insert report HTML directly into the page
                    const reportDiv = document.createElement('div');
                    reportDiv.id = 'reportContent';
                    reportDiv.innerHTML = bodyContent;
                    reportDiv.style.marginTop = '20px';
                    
                    document.getElementById('results').appendChild(reportDiv);
                    
                    // Wait a moment for DOM to settle, then execute scripts
                    setTimeout(() => {
                        const scripts = reportDoc.querySelectorAll('script');
                        scripts.forEach(script => {
                            // Skip external script tags (like Chart.js which is already loaded)
                            if (!script.src) {
                                const newScript = document.createElement('script');
                                newScript.textContent = script.textContent;
                                newScript.async = false;
                                document.body.appendChild(newScript);
                            }
                        });
                    }, 100);  // Small delay to ensure DOM is ready
                    
                    document.getElementById('results').style.display = 'block';
                    document.getElementById('reportsSection').style.display = 'none';
                } else {
                    throw new Error(data.error || 'Analysis failed');
                }
            } catch (err) {
                document.getElementById('errorAlert').textContent = 'Error: ' + err.message;
                document.getElementById('errorAlert').style.display = 'block';
                document.getElementById('reportsSection').style.display = 'flex';
                document.getElementById('results').style.display = 'none';
            } finally {
                document.getElementById('loading').style.display = 'none';
                analyzeBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Home page."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze comments and generate report."""
    try:
        # Load models on first request (lazy loading)
        _load_models()
        
        data = request.json
        url = data.get("url")
        max_comments = int(data.get("maxComments", 0))
        use_context = data.get("useContext", False) and CONTEXT_AVAILABLE and context_pipeline is not None
        
        # Generate report number based on existing reports
        import glob
        report_files = glob.glob(os.path.join(REPORT_DIR, "toxicity_report_*.html"))
        report_number = len(report_files) + 1
        title = f"Toxicity Report #{report_number}"
        
        if not url:
            return jsonify({"success": False, "error": "URL is required"})
        
        # Fetch comments
        print(f"Fetching comments from {url}...")
        print(f"[DEBUG] Max comments requested: {max_comments}")
        
        # If max_comments is 0 or not specified, use default
        if max_comments <= 0:
            max_comments = 100000
            print(f"[DEBUG] Using default max_comments: {max_comments}")
        
        try:
            comments = fetcher.fetch_comments(url, max_comments)
        except ValueError as e:
            # likely missing API key or unsupported URL
            msg = str(e)
            if "API key" in msg:
                msg += ". Be sure to set YOUTUBE_API_KEY in .env or environment and restart the app."
            return jsonify({"success": False, "error": msg})
        except Exception as e:
            # Any other error from comment fetching (API errors, network, disabled comments, etc)
            error_msg = str(e)
            return jsonify({"success": False, "error": error_msg})
        
        if not comments:
            return jsonify({
                "success": False,
                "error": "No comments found. Check URL or API credentials."
            })
        
        # Analyze with context or standard
        print(f"Analyzing {len(comments)} comments (context={use_context})...")
        
        if use_context:
            # Use context-aware analysis
            df = analyze_comments_with_context(comments, context_pipeline)
        else:
            # Use standard analysis
            df = analyze_comments(comments, analyzer)
        
        # Generate report and get HTML content
        print("Generating report...")
        report_path = report_gen.generate_report(df, url, title)
        
        # Get the HTML content of the report
        with open(report_path, 'r', encoding='utf-8') as f:
            report_html = f.read()
        
        # Calculate stats
        stats = {
            "total_comments": len(df),
            "toxic_comments": int(df["is_toxic"].sum()),
            "toxic_rate": float(100 * df["is_toxic"].sum() / len(df)) if len(df) > 0 else 0,
            "avg_confidence": float(df["max_confidence"].mean()),
            "report_html": report_html,  # Include full HTML in response
            "use_context": use_context
        }
        
        return jsonify({
            "success": True,
            "report_path": f"/download/{os.path.basename(report_path)}",
            **stats
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


def analyze_comments_with_context(comments: list, context_pipeline) -> pd.DataFrame:
    """
    Analyze comments using conversation context awareness.
    
    Groups comments by thread and analyzes them with context from previous comments.
    """
    results = []
    
    # Create a thread structure from comments
    # If comments don't have thread_id, assign them sequentially
    comments_with_ids = []
    for i, comment in enumerate(comments):
        if isinstance(comment, dict):
            comment_obj = {
                'comment_id': f"c_{i}",
                'text': comment.get('text', ''),
                'author': comment.get('author', 'Unknown'),
                'thread_id': 't_0'  # Group all in single thread for YouTube
            }
        else:
            comment_obj = {
                'comment_id': f"c_{i}",
                'text': str(comment),
                'author': 'Unknown',
                'thread_id': 't_0'
            }
        comments_with_ids.append(comment_obj)
    
    # Predict with context
    try:
        predictions = context_pipeline.predict_thread(comments_with_ids, use_context=True)
    except Exception as e:
        print(f"[WARNING] Context-aware prediction failed, falling back to standard: {e}")
        return analyze_comments(comments, analyzer)
    
    # Convert to DataFrame format compatible with report generator
    for pred in predictions:
        result = {
            'text': pred.get('original_text', pred.get('text', '')),
            'author': pred.get('author', 'Unknown'),
            'is_toxic': pred.get('is_toxic', False),
            'max_confidence': pred.get('confidence', 0.5),
            'context_used': pred.get('context_used', False),
            'context_size': pred.get('context_size', 0)
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    return df


# determine report directory same as report generator
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

@app.route("/download/<filename>")
def download(filename):
    """Download report."""
    path = os.path.join(REPORT_DIR, filename)
    if not os.path.exists(path):
        # maybe someone requested from wrong cwd - try relative
        path = os.path.join(os.getcwd(), "reports", filename)
    return send_file(
        path,
        as_attachment=True,
        download_name=filename
    )


if __name__ == "__main__":
    init_services()
    # Disable auto-reload to prevent interruptions during model inference
    # (torch operations were triggering false positives for file changes)
    app.run(debug=True, port=5000, use_reloader=False)
