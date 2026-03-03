# Vibe Check : Cyberbulling Detection System

## Overview
VibeCheck is a web-based tool designed to analyze YouTube video comments in real-time. It uses Machine Learning to identify toxic behavior, harassment, and cyberbullying, providing creators and viewers with a "vibe" report of the digital environment.

## Core Features

### URL Input and Comment Fetching
- Input field for YouTube video or post URLs
- Integration with YouTube Data API v3 to fetch comments from the provided URL
- Display fetched comments in a readable format

### Toxicity Analysis
- Integration with local toxicity detection models (e.g. Hugging Face `unitary/toxic-bert`)
- Analyze each fetched comment for toxicity levels using the selected API
- Color-coded display system for toxicity levels:
  - Low toxicity (green)
  - Medium toxicity (yellow/orange)
  - High toxicity (red)

### API Configuration
- Modal interface for users to input and store their API keys
- Support for local Hugging Face model configuration
- Secure storage of API credentials for toxicity analysis

### Reporting and Flagging
- "Report" button for each comment that uses YouTube API's reporting features
- "Flag" button for each comment to mark inappropriate content
- Send feedback to YouTube through official reporting mechanisms

### Toxicity Report Generation
- Generate comprehensive toxicity reports summarizing detected bullying
- Include statistics on toxicity levels found
- Downloadable report in a standard format (PDF or text)
- Report contains comment analysis summary and recommendations

## Backend Data Storage
- Store analysis results and toxicity scores for generated reports
- Save user-generated reports for download functionality
- Cache YouTube comment data temporarily to avoid redundant API calls
- Store user API key configurations securely

## Backend Operations
- Fetch comments from YouTube Data API v3
- Process comments through the local toxicity detection model
- Normalize toxicity scores and severity levels to consistent format ("low", "medium", "high")
- Handle YouTube API reporting and flagging requests
- Generate and store toxicity reports
- Manage API rate limits and error handling for multiple API providers

## User Interface
- Clean, intuitive interface for URL input
- Comment display with clear toxicity indicators
- Easy-to-use report and flag buttons
- API configuration modal for key management
- Report generation and download interface
- Responsive design for various screen sizes

## Language
- Application content and interface in English
