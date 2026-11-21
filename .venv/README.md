# Social Media Sentiment Analysis & FastAPI Backend

An end-to-end Natural Language Processing (NLP) pipeline and REST API that analyzes public sentiment across social media posts, visualizes opinion trends, and serves real-time predictions via FastAPI.

## Features
- **NLP Vectorization:** Cleans and tokenizes text using `TfidfVectorizer`.
- **Sentiment Classification:** Trains a `MultinomialNB` (Naive Bayes) classifier across `Positive`, `Negative`, and `Neutral` categories.
- **Trend Visualization:** Generates bar chart visualizations of sentiment distribution with `Matplotlib`.
- **FastAPI Backend:** Exposes `POST /predict` for custom text classification and `GET /live-analysis/{keyword}` for keyword stream analysis.

## How to Run Locally

1. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn nltk fastapi uvicorn matplotlib tweepy

   ## Sample Output

![Public Opinion Sentiment Trends for AI](Figure_1.png)