"""
Generate toxicity reports with visualizations.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List
from datetime import datetime
import os


class ToxicityReportGenerator:
    """Generate HTML reports showing toxicity analysis results."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Args:
            output_dir: Directory to save reports (relative to project root)
        """
        # compute absolute path based on project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        abs_dir = os.path.join(project_root, output_dir)
        self.output_dir = abs_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_report(self, df: pd.DataFrame, url: str, title: str = None) -> str:
        """
        Generate HTML report from analyzed comments DataFrame.
        
        Args:
            df: DataFrame from analyze_comments() with toxicity analysis
            url: Video/post URL being analyzed
            title: Optional title for the report
        
        Returns:
            Path to generated HTML report
        """
        if title is None:
            title = f"Toxicity Report - {url}"
        
        # Calculate statistics
        stats = self._calculate_stats(df)
        
        # Generate HTML
        html = self._build_html(df, url, title, stats)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"toxicity_report_{timestamp}.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"Report saved to {report_path}")
        return report_path
    
    def _calculate_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics."""
        total_comments = len(df)
        toxic_comments = df["is_toxic"].sum()
        toxic_rate = 100 * toxic_comments / total_comments if total_comments > 0 else 0
        
        # Severity breakdown
        severity_counts = df["severity"].value_counts().to_dict()
        
        # Label breakdown
        all_labels = []
        for labels_list in df["toxic_labels"]:
            all_labels.extend(labels_list)
        label_counts = pd.Series(all_labels).value_counts().to_dict()
        
        # Average confidence
        avg_confidence = df["max_confidence"].mean()
        
        # Top toxic comments
        toxic_df = df[df["is_toxic"]].sort_values("max_confidence", ascending=False)
        top_toxic = toxic_df.head(10)[["text", "toxic_labels", "severity", "max_confidence"]].to_dict("records")
        
        return {
            "total_comments": total_comments,
            "toxic_comments": toxic_comments,
            "toxic_rate": toxic_rate,
            "severity_counts": severity_counts,
            "label_counts": label_counts,
            "avg_confidence": avg_confidence,
            "top_toxic": top_toxic
        }
    
    def _build_html(self, df: pd.DataFrame, url: str, title: str, stats: Dict) -> str:
        """Build HTML report content."""
        
        # Severity color mapping
        severity_colors = {0: "#4CAF50", 1: "#FF9800", 2: "#FF5722", 3: "#F44336"}  # Safe, Mild, Moderate, Severe
        severity_labels_map = {0: "Safe", 1: "Mild", 2: "Moderate", 3: "Severe"}
        
        # Build label chart data
        label_data = stats["label_counts"]
        label_chart = self._build_chart_data("Toxicity Types", label_data, "bar")
        
        # Build severity chart data
        severity_data = stats["severity_counts"]
        severity_data = {severity_labels_map.get(k, str(k)): v for k, v in severity_data.items()}
        severity_chart = self._build_chart_data("Severity Distribution", severity_data, "doughnut")
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        h1 {{ 
            color: #333; 
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card h3 {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .stat-card .percentage {{
            font-size: 12px;
            opacity: 0.8;
            margin-top: 5px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}
        .chart-container {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #eee;
        }}
        .chart-container h3 {{
            margin-bottom: 15px;
            color: #333;
        }}
        .top-toxic {{
            background: #fff3e0;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
            margin-bottom: 30px;
        }}
        .top-toxic h3 {{
            color: #e65100;
            margin-bottom: 15px;
        }}
        .comment-item {{
            background: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 6px;
            border-left: 4px solid #f44336;
        }}
        .comment-item .text {{
            font-style: italic;
            color: #333;
            margin-bottom: 8px;
        }}
        .comment-item .meta {{
            display: flex;
            gap: 15px;
            font-size: 12px;
            color: #999;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            margin-right: 5px;
        }}
        .badge-toxic {{ background: #ffebee; color: #c62828; }}
        .badge-threat {{ background: #fce4ec; color: #c2185b; }}
        .badge-severe {{ background: #f3e5f5; color: #7b1fa2; }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #999;
            font-size: 12px;
        }}
        .severity-0 {{ color: #4CAF50; }}
        .severity-1 {{ color: #FF9800; }}
        .severity-2 {{ color: #FF5722; }}
        .severity-3 {{ color: #F44336; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚨 Toxicity Analysis Report</h1>
        <div class="subtitle">
            Source: <strong>{url}</strong><br>
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        
        <!-- Statistics Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Comments</h3>
                <div class="value">{stats['total_comments']}</div>
            </div>
            <div class="stat-card">
                <h3>Toxic Comments</h3>
                <div class="value">{stats['toxic_comments']}</div>
                <div class="percentage">{stats['toxic_rate']:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Safe Comments</h3>
                <div class="value">{stats['total_comments'] - stats['toxic_comments']}</div>
                <div class="percentage">{100 - stats['toxic_rate']:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Avg Confidence</h3>
                <div class="value">{stats['avg_confidence']:.2%}</div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="charts-grid">
            <div class="chart-container">
                <h3>📊 Toxicity Type Distribution</h3>
                <canvas id="labelChart" height="80"></canvas>
            </div>
            <div class="chart-container">
                <h3>⚠️ Severity Breakdown</h3>
                <canvas id="severityChart" height="80"></canvas>
            </div>
        </div>
        
        <!-- Top Toxic Comments -->
        <div class="top-toxic">
            <h3>🔴 Top {min(10, len(stats['top_toxic']))} Most Toxic Comments</h3>
            {''.join([self._build_comment_item(c, severity_colors, severity_labels_map) for c in stats['top_toxic']])}
        </div>
    </div>
    
    <script>
        // Label Distribution Chart
        const labelCtx = document.getElementById('labelChart').getContext('2d');
        new Chart(labelCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(list(stats['label_counts'].keys()))},
                datasets: [{{
                    label: 'Count',
                    data: {json.dumps(list(stats['label_counts'].values()))},
                    backgroundColor: 'rgba(102, 126, 234, 0.7)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});
        
        // Severity Chart
        const severityCtx = document.getElementById('severityChart').getContext('2d');
        new Chart(severityCtx, {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(list(severity_data.keys()))},
                datasets: [{{
                    data: {json.dumps(list(severity_data.values()))},
                    backgroundColor: [
                        '#4CAF50',
                        '#FF9800', 
                        '#FF5722',
                        '#F44336'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true
            }}
        }});
    </script>
</body>
</html>
        """
        return html
    
    def _build_comment_item(self, comment: Dict, severity_colors: Dict, severity_labels_map: Dict) -> str:
        """Build HTML for a single comment."""
        severity = comment["severity"]
        severity_label = severity_labels_map.get(severity, str(severity))
        severity_color = severity_colors.get(severity, "#999")
        
        labels_html = "".join([
            f'<span class="badge badge-{label.replace("_", "-")}">{label}</span>'
            for label in comment["toxic_labels"]
        ])
        
        return f"""
        <div class="comment-item">
            <div class="text">"{comment['text'][:200]}{'...' if len(comment['text']) > 200 else ''}"</div>
            <div class="meta">
                <span>Labels: {labels_html if labels_html else '<em>generic toxicity</em>'}</span>
                <span>Severity: <strong style="color: {severity_color}">{severity_label}</strong></span>
                <span>Confidence: {comment['max_confidence']:.1%}</span>
            </div>
        </div>
        """
    
    def _build_chart_data(self, title: str, data: Dict, chart_type: str) -> str:
        """Build chart data JSON."""
        return json.dumps({
            "title": title,
            "type": chart_type,
            "labels": list(data.keys()),
            "data": list(data.values())
        })


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(".")
    from toxicity_analyzer import ToxicityAnalyzer
    
    # Create sample data
    sample_comments = [
        {"text": "I love this!", "toxic_labels": [], "severity": 0, "is_toxic": False, "max_confidence": 0.1},
        {"text": "You're stupid", "toxic_labels": ["insult"], "severity": 1, "is_toxic": True, "max_confidence": 0.8},
        {"text": "I hope you die", "toxic_labels": ["threat"], "severity": 3, "is_toxic": True, "max_confidence": 0.95},
    ]
    df = pd.DataFrame(sample_comments)
    
    generator = ToxicityReportGenerator()
    report_path = generator.generate_report(df, "https://youtube.com/watch?v=example", "Test Report")
    print(f"Report generated: {report_path}")
