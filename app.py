"""
app.py - FastAPI backend for Social Media Sentiment Analysis.

Run:  uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs

Optional live data: export TWITTER_BEARER_TOKEN=<your X/Twitter API v2 bearer token>
Without it (or if the API call fails) a built-in simulator is used.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from sentiment_analyzer import LABELS, SentimentModel, clean_text, simulate_live_posts, train_model

state: Dict[str, SentimentModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Trains on startup and prints accuracy + classification report.
    state["model"] = train_model(verbose=True)
    yield
    state.clear()


app = FastAPI(
    title="Social Media Sentiment Analysis API",
    description=(
        "TF-IDF + Multinomial Naive Bayes sentiment classifier (Positive / Negative / Neutral) "
        "with a single-text prediction endpoint and a bulk live-analysis endpoint backed by "
        "Tweepy (Twitter/X API v2) or a built-in simulator."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_model() -> SentimentModel:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    return model


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, examples=["I love the new update, it's so smooth!"])


class Probabilities(BaseModel):
    negative: float
    neutral: float
    positive: float


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Probabilities


class PostPrediction(BaseModel):
    text: str
    created_at: Optional[str] = None
    sentiment: str
    confidence: float


class LiveAnalysisResponse(BaseModel):
    keyword: str
    data_source: str
    total_posts: int
    sentiment_counts: Dict[str, int]
    sentiment_percentages: Dict[str, float]
    dominant_sentiment: Optional[str]
    net_sentiment_score: float = Field(..., description="(positive - negative) / total, range -1..1")
    average_confidence: float
    trend_summary: str
    sample_posts: List[PostPrediction]


# --------------------------------------------------------------------------- #
# Post fetching (Tweepy API v2 with simulated fallback)
# --------------------------------------------------------------------------- #
def fetch_recent_posts(keyword: str, limit: int) -> Tuple[List[Dict], str]:
    token = os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        return simulate_live_posts(keyword, limit), "simulated"
    try:
        import tweepy

        client = tweepy.Client(bearer_token=token)
        safe_kw = keyword.replace('"', "")
        response = client.search_recent_tweets(
            query=f'"{safe_kw}" -is:retweet lang:en',
            max_results=max(10, min(limit, 100)),  # API v2 allows 10..100
            tweet_fields=["created_at", "lang"],
        )
        tweets = response.data or []
        posts = [
            {
                "id": str(t.id),
                "text": t.text,
                "created_at": t.created_at.isoformat() if getattr(t, "created_at", None) else None,
            }
            for t in tweets[:limit]
        ]
        return posts, "twitter_api_v2"
    except Exception as exc:  # network error, bad token, rate limit...
        return simulate_live_posts(keyword, limit), f"simulated (twitter fallback: {type(exc).__name__})"


def describe_trend(score: float, dominant: str) -> str:
    if score >= 0.35:
        mood = "strongly positive"
    elif score >= 0.10:
        mood = "leaning positive"
    elif score <= -0.35:
        mood = "strongly negative"
    elif score <= -0.10:
        mood = "leaning negative"
    else:
        mood = "mixed / neutral"
    return f"Public opinion is {mood} (net score {score:+.2f}); most common sentiment: {dominant}."


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", tags=["Health"], summary="Health check")
def root(model: SentimentModel = Depends(get_model)):
    return {
        "status": "ok",
        "model": "TF-IDF + MultinomialNB",
        "test_accuracy": model.metrics["accuracy"],
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"], summary="Predict sentiment of one text")
def predict(payload: PredictRequest, model: SentimentModel = Depends(get_model)):
    if not clean_text(payload.text):
        raise HTTPException(status_code=422, detail="Text contains no analysable words after cleaning.")
    return model.predict_one(payload.text)


@app.get(
    "/live-analysis/{keyword}",
    response_model=LiveAnalysisResponse,
    tags=["Live analysis"],
    summary="Fetch recent posts for a keyword and summarise public opinion",
)
def live_analysis(
    keyword: str = Path(..., min_length=1, max_length=100, description="Topic, product or hashtag"),
    limit: int = Query(50, ge=10, le=100, description="Number of recent posts to analyse"),
    model: SentimentModel = Depends(get_model),
):
    keyword = keyword.strip()
    if not keyword:
        raise HTTPException(status_code=422, detail="Keyword must not be blank.")

    posts, source = fetch_recent_posts(keyword, limit)
    if not posts:
        return LiveAnalysisResponse(
            keyword=keyword, data_source=source, total_posts=0,
            sentiment_counts={l: 0 for l in LABELS}, sentiment_percentages={l: 0.0 for l in LABELS},
            dominant_sentiment=None, net_sentiment_score=0.0, average_confidence=0.0,
            trend_summary="No recent posts found for this keyword.", sample_posts=[],
        )

    predictions = model.predict_batch([p["text"] for p in posts])
    counts = {l: 0 for l in LABELS}
    for pred in predictions:
        counts[pred["sentiment"]] += 1
    total = len(predictions)
    percentages = {l: round(100 * c / total, 2) for l, c in counts.items()}
    dominant = max(counts, key=counts.get)
    net = round((counts["Positive"] - counts["Negative"]) / total, 4)
    avg_conf = round(sum(p["confidence"] for p in predictions) / total, 4)

    samples = [
        PostPrediction(
            text=pred["text"], created_at=post.get("created_at"),
            sentiment=pred["sentiment"], confidence=pred["confidence"],
        )
        for post, pred in list(zip(posts, predictions))[:10]
    ]
    return LiveAnalysisResponse(
        keyword=keyword, data_source=source, total_posts=total,
        sentiment_counts=counts, sentiment_percentages=percentages,
        dominant_sentiment=dominant, net_sentiment_score=net, average_confidence=avg_conf,
        trend_summary=describe_trend(net, dominant), sample_posts=samples,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
