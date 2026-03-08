# VibeCheck - Cyberbullying Detection System

> VibeCheck is a web-based tool designed to analyze YouTube video comments in real-time. It uses an optimized Machine Learning model (DistilBERT with fine-tuning) to identify toxic behavior, harassment, and cyberbullying, providing creators and viewers with a comprehensive "vibe" report of the digital environment.

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

## � Quick Start

### Prerequisites
- Python 3.12
- YouTube API key ([Get one here](https://developers.google.com/youtube/registering_an_application))
- 256 MB RAM minimum

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
