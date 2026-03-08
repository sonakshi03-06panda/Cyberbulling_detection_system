# VibeCheck - Cyberbullying Detection System

> VibeCheck is a web-based tool designed to analyze YouTube post comments in real-time. It uses an optimized Machine Learning model (DistilBERT with fine-tuning) to identify toxic behavior, harassment, and cyberbullying, providing creators and viewers with a comprehensive "vibe" report of the digital environment.

**Status**: ✅ Production Ready (v2.2)  
**Last Updated**: March 2026

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 82.15% |
| **Processing Speed** | ~127 samples/sec |
| **Model Size** | 89 MB |
| **Memory Usage** | 256 MB |
| **Comment Capacity** | 100,000/video |
| **Status** | Production Deployed |

## ⭐ Features

### Toxicity Detection
- **DistilBERT** fine-tuned model (66M parameters)
- **6 toxicity categories**: Toxic, Severe, Obscene, Threats, Insults, Hate Speech
- **Context-aware**: Detects sarcasm, emoji meaning, ALL CAPS context
- **Confidence scores**: Per-prediction confidence metrics

### Comment Analysis
- Fetch up to **100,000** comments per YouTube video
- Automatic **nested reply extraction** (up to 5 levels)
- Smart **deduplication** and cleaning
- **Batch processing** (128 comments/batch) for speed

### Reporting
- **HTML reports** with visualizations
- Diversity-based examples by category type
- Safe comments included for context
- Charts: Toxicity types & severity distribution
- Export-ready format

### Interfaces
- **Flask Web UI** (Port 5000) - Main dashboard
- **FastAPI REST API** (Port 8000) - 12 endpoints
- **Streamlit Dashboard** (Port 8501) - Analytics & monitoring

## 💻 TECH STACK

### Machine Learning Stack
```
Framework                Version
─────────────────────────────────
PyTorch                  2.2.2
Hugging Face Transform.  4.40.1
DistilBERT Model         Fine-tuned
SHAP (Explainability)    0.46.0
scikit-learn             1.4.2
```

### Backend & API
```
Framework                Version      Purpose
──────────────────────────────────────────────
Flask                    2.3.2        Main web app
FastAPI                  0.104.1      Analytics REST API
Uvicorn                  0.24.0       ASGI server
```

### Frontend & Visualization
```
Framework                Version      Purpose
──────────────────────────────────────────────
Streamlit                1.28.1       Dashboard UI
Plotly                   5.17.0       Interactive charts
HTML/CSS/JS              Native       Web interface
```

### Data & Analysis
```
Library                  Version      Purpose
──────────────────────────────────────────────
Pandas                   2.2.2        Data manipulation
NumPy                    1.26.4       Numerical computing
Matplotlib               3.8.4        Static visualizations
SciPy                    1.11.2       Scientific computing
```

### External Integrations
```
Service                  Version      Purpose
──────────────────────────────────────────────
YouTube Data API v3      2.126.0      Comment fetching
```

### Utilities
```
Library                  Version      Purpose
──────────────────────────────────────────────
python-dotenv            1.0.0        Environment config
emoji                    2.11.0       Emoji handling
tqdm                     4.66.4       Progress bars
Pydantic                 2.5.0        Data validation
requests                 2.31.0       HTTP client
```

### Development Stack
```
Python Version: 3.12.x
OS Support: Windows, macOS, Linux
Package Manager: pip
Virtual Environment: venv/virtualenv
```

## 📁 Project Structure

```
scr/
├── app.py                    # Flask web interface
├── toxicity_analyzer.py      # ML model inference
├── comment_fetcher.py        # YouTube API integration
├── report_generator.py       # HTML report generation
├── preprocessing.py          # Text preprocessing
├── analytics_api.py          # FastAPI backend
└── dashboard.py              # Streamlit dashboard

models/
├── final_model/              # DistilBERT fine-tuned
└── final_model_quantized/    # 8-bit quantized version

data/
├── train.csv                 # Training dataset
└── test.csv                  # Test dataset
```

## 🔗 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/analyze` | Analyze YouTube video |
| GET | `/analytics/summary` | Get summary stats |
| GET | `/analytics/timeline` | Get hourly trends |
| GET | `/download/<file>` | Download report |


## 🤝 Contributing

Found a bug? Have a suggestion? Open an issue or submit a pull request!

---

**Built with** PyTorch • Hugging Face • Flask • FastAPI • Streamlit