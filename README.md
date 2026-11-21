# Social Media Sentiment Analysis & FastAPI Backend

An end-to-end Natural Language Processing (NLP) pipeline and REST API that processes social media text, classifies sentiment (`Positive`, `Negative`, and `Neutral`) using TF-IDF and Multinomial Naive Bayes, visualizes opinion trends with Matplotlib, and serves real-time predictions via FastAPI.

## Features

- **NLP Preprocessing & Vectorization:** Cleans, tokenizes, and transforms text using `TfidfVectorizer` (lowercasing, stop-word removal, and TF-IDF weighting).
- **Naive Bayes Classifier:** Trains a `MultinomialNB` model to predict sentiment labels along with per-class confidence probabilities.
- **Trend Visualization:** Generates a bar chart of sentiment distribution with `Matplotlib`.
- **FastAPI Backend & Swagger UI:** Provides interactive API documentation at `/docs` along with endpoints for custom text inference and keyword-based stream analysis.

## Project Structure

```text
social-media-sentiment-analysis/
├── app.py                  # FastAPI application with /predict and /live-analysis/{keyword}
├── sentiment_analyzer.py   # Model training, evaluation metrics, and Matplotlib visualization
├── .gitignore              # Ignored virtual environments, cache files, and secrets
└── README.md               # Project documentation
