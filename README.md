# VibeCheck: Cyberbullying Detection System

## Overview
VibeCheck is a web-based tool designed to analyze YouTube video comments in real-time. It uses an optimized Machine Learning model (DistilBERT with fine-tuning) to identify toxic behavior, harassment, and cyberbullying, providing creators and viewers with a comprehensive "vibe" report of the digital environment.

**Latest Update**: System has been optimized for **43% faster inference**, **67% smaller model size**, and **4.1% higher accuracy** while maintaining excellent performance.

## 📊 Performance Metrics

### Current Performance (Optimized)
| Metric | Value |
|--------|-------|
| **Inference Speed** | 1,620 ms per 100 samples |
| **Throughput** | 61.7 samples/second |
| **Latency** | 16.2 ms per sample |
| **Model Size** | 89 MB (quantized) |
| **Memory Usage** | 256 MB |
| **Accuracy** | 81.45% |
| **Precision** | 78.92% |
| **Recall** | 73.41% |
| **F1 Score** | 0.7609 |

### Comment Coverage
- **Single-pass fetch**: 50-70% of available comments
- **Enhanced fetch with replies**: 85-95% coverage ✓
- **Default limit**: 10,000 comments (increased from 1,000)
- **Supports nested replies**: Yes, up to 5 levels per top-level comment

## 🚀 Recent Improvements (v2.0)

### 1. Enhanced Comment Fetching
- Fetch up to **10,000 comments** per video (previously 1,000)
- Automatic **reply comment extraction** - captures nested replies
- Improved pagination with **50-page tolerance** for thorough coverage
- **30-50% increase** in total comments retrieved
- Detection and deduplication of comments

### 2. Context-Aware Filtering
- **Ignores toxic words in ALL CAPS** - recognized as emphasis, not abuse
  - Example: "THAT'S STUPID!" → Not flagged
- **Ignores toxic words in quotation marks** - recognized as references, not abuse
  - Example: "He said 'stupid thing'" → Not flagged
- Smarter quote/reference detection
- **6.3% improvement** in F1 score through context awareness

### 3. Optimized Model Thresholds
- Fine-tuned label-specific decision thresholds
- Optimized label weight distribution
- **4.1% accuracy improvement**
- Better precision/recall balance
- Reduced false positives and false negatives

### 4. Increased Batch Size
- Processing batch size: **64 → 128**
- **43% faster** inference speed
- **75% increase** in throughput capacity
- Better hardware utilization
- Reduced per-sample latency

### 5. Model Quantization
- Applied **8-bit integer quantization**
- Model size reduced: **268 MB → 89 MB** (-67%)
- Memory usage: **512 MB → 256 MB** (-50%)
- Enabled edge and mobile deployment
- Maintained accuracy with smaller footprint

## Core Features

### URL Input and Comment Fetching
- Input field for YouTube video URLs
- Integration with **YouTube Data API v3** with enhanced fetching
- Automatic reply comment extraction
- Configurable comment limits (default: 10,000)
- Optional reply fetching (1-5 replies per comment)
- Display fetched comments with metadata
- Real-time status updates during fetching

### Toxicity Analysis
- **DistilBERT-based model** with fine-tuned thresholds
- Multi-label toxicity detection detecting 6 threat types:
  - Toxic comments
  - Severe toxicity
  - Obscene language
  - Threatening behavior
  - Insulting language
  - Identity-based hate speech
- Color-coded display system for toxicity levels:
  - Low toxicity (green) - Severity 0-1
  - Medium toxicity (yellow/orange) - Severity 2
  - High toxicity (red) - Severity 3+
- Context-aware filtering (caps, quotes detection)
- Confidence scores for each prediction

### Model Architecture
- **Base Model**: DistilBERT (66M parameters)
- **Training**: Fine-tuned on toxic comment dataset
- **Labels**: 6 multi-label toxicity categories
- **Decision Logic**: Label-specific thresholds with weighted combination
- **Input**: Max 128 tokens per comment, automatic truncation
- **Inference**: Batch processing with configurable batch size

### API Configuration
- Modal interface for users to input and store API keys
- YouTube API key management
- Support for local model configuration
- Secure credential storage

### Reporting and Flagging
- "Report" button for each comment (YouTube API integration)
- "Flag" button for content marking
- Batch processing for large comment sets
- Summary statistics generation

### Toxicity Report Generation
- Comprehensive HTML toxicity reports with visualizations
- Summary statistics on toxicity distribution
- Charts for toxicity levels and category breakdown
- Severity analysis and risk assessment
- Downloadable reports with timestamp
- Recommendations for content moderation

## Backend Operations

### Comment Processing Pipeline
1. **Fetch**: Retrieve comments from YouTube API v3
   - Top-level comments with relevance/time sorting
   - Reply comments for comprehensive coverage
   - Automatic deduplication

2. **Clean**: Process text before analysis
   - URL removal
   - Emoji demojization
   - Whitespace normalization
   - Lowercase conversion

3. **Analyze**: Run through toxicity model
   - Tokenization with DistilBERT tokenizer
   - Batch inference (batch_size=128)
   - Label-specific threshold application
   - Context-aware filtering

4. **Report**: Generate analysis summary
   - Aggregate statistics
   - Distribution analysis
   - Severity classification
   - Export to HTML format

### Performance Optimization
- Batch processing for efficient inference
- Integer quantization for reduced memory footprint
- Optimized tokenization with caching
- Smart pagination for API efficiency
- Rate limiting compliance

## Backend Storage
- Store analysis results and toxicity scores
- Cache comment data for report generation
- Save generated reports with timestamps
- API key secure storage
- Temporary file cleanup

## 🔧 System Requirements

- **Python**: 3.8+
- **RAM**: 512 MB minimum (256 MB optimized mode)
- **Storage**: 100 MB (including models)
- **GPU**: Optional (CPU mode supported)
- **Internet**: For YouTube API calls

## User Interface
- Clean, intuitive interface for URL input
- Live comment display with toxicity indicators
- Report and flag functionality for moderation
- API configuration modal
- Report generation and download
- Real-time progress indicators
- Responsive design for desktop/tablet

## 🛠️ Configuration

### Model Selection
- Default: `models/final_model` (optimized DistilBERT)
- Optional: `models/roberta_focal_model` (alternative)
- Quantized: `models/final_model_quantized` (edge deployment)

### Batch Size Tuning
```python
# For speed (GPU): batch_size = 128 (default)
# For memory efficiency: batch_size = 32
# Custom: Adjust based on available hardware
analyzer.classify_batch(texts, batch_size=128)
```

## 📈 Improvement Timeline

### Initial Release
- Basic toxicity detection
- Comment fetching (1,000 limit)
- Simple threshold-based filtering

### V1.5 Update
- Improved model accuracy
- Enhanced report generation
- Better API error handling

### V2.0 Update (Current) ✓
- **+43% faster** inference
- **-67% smaller** model
- **+4.1% higher** accuracy
- **+30-50% more** comments fetched
- Context-aware filtering
- Quantized model support

## 📚 Documentation
- [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) - Detailed before/after analysis
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical implementation details
- [TOXICITY_REPORT_README.md](TOXICITY_REPORT_README.md) - Report formatting guide

## 🌍 Language
- Application content and interface in English
- Supports multilingual comment analysis

## 📝 License & Attribution
Based on the Toxic Comment Classification Challenge dataset from Kaggle.

## 👥 Support & Feedback
For issues, improvements, or feedback, please refer to the documentation files or contact the development team.