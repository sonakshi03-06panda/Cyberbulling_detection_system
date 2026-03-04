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
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from comment_fetcher import CommentFetcher
from toxicity_analyzer import ToxicityAnalyzer, analyze_comments
from report_generator import ToxicityReportGenerator

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max


# Initialize analyzers
analyzer = None
fetcher = None
report_gen = ToxicityReportGenerator("reports")


def init_services():
    """Initialize services."""
    global analyzer, fetcher
    analyzer = ToxicityAnalyzer("models/final_model")
    # Initialize fetcher with API key from environment
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    print(f"Loaded YOUTUBE_API_KEY: {youtube_key}")
    if not youtube_key:
        # we'll still create the object so error is returned later,
        # but log a warning for clarity
        print("WARNING: YOUTUBE_API_KEY not set. comment fetching will fail.")
    fetcher = CommentFetcher(youtube_key)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vibe Check - YouTube Safety Monitor</title>
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
                <h2>Analyze YouTube Video</h2>
                
                <div class="alert alert-error" id="errorAlert"></div>
                <div class="alert alert-success" id="successAlert"></div>
                
                <form id="reportForm">
                    <div class="form-group">
                        <label for="url">Enter a YouTube video URL to analyze comments for cyberbullying and toxic content</label>
                        <input 
                            type="url" 
                            id="url" 
                            name="url" 
                            placeholder="https://www.youtube.com/watch?v=..." 
                            required
                        >
                    </div>
                    
                    <div class="form-group">
                        <label for="maxComments">Max Comments to Analyze</label>
                        <input 
                            type="number" 
                            id="maxComments" 
                            name="maxComments" 
                            value="500" 
                            min="10" 
                            max="5000"
                        >
                    </div>
                    
                    <div class="form-group">
                        <label for="title">Report Title (Optional)</label>
                        <input 
                            type="text" 
                            id="title" 
                            name="title" 
                            placeholder="e.g., My Video Analysis"
                        >
                    </div>
                    
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
            const title = document.getElementById('title').value;
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
                    body: JSON.stringify({ url, maxComments, title })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('summary').innerHTML = `
                        <strong>Analysis Complete!</strong><br><br>
                        <span class="badge badge-info">${data.total_comments} Comments</span>
                        <span class="badge badge-danger">${data.toxic_comments} Toxic (${data.toxic_rate.toFixed(1)}%)</span>
                        <span class="badge badge-warning">Confidence: ${(data.avg_confidence * 100).toFixed(1)}%</span>
                    `;
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
        data = request.json
        url = data.get("url")
        max_comments = int(data.get("maxComments", 500))
        title = data.get("title", f"Toxicity Report - {url}")
        
        if not url:
            return jsonify({"success": False, "error": "URL is required"})
        
        # Fetch comments
        print(f"Fetching comments from {url}...")
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
        
        # Analyze
        print(f"Analyzing {len(comments)} comments...")
        df = analyze_comments(comments)
        
        # Generate report
        print("Generating report...")
        report_path = report_gen.generate_report(df, url, title)
        
        # Calculate stats
        stats = {
            "total_comments": len(df),
            "toxic_comments": int(df["is_toxic"].sum()),
            "toxic_rate": float(100 * df["is_toxic"].sum() / len(df)) if len(df) > 0 else 0,
            "avg_confidence": float(df["max_confidence"].mean())
        }
        
        return jsonify({
            "success": True,
            "report_path": f"/download/{os.path.basename(report_path)}",
            **stats
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)})


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
    app.run(debug=True, port=5000)
