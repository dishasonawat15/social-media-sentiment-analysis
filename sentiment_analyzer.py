"""
sentiment_analyzer.py
=====================
Core NLP / ML pipeline AND standalone visualization script.

Used in two ways:
  1. Imported by app.py (dataset simulation, preprocessing, model training,
     prediction helpers, simulated live-post stream).
  2. Run directly to train the model, print metrics and plot public-opinion
     sentiment trends with matplotlib:

         python sentiment_analyzer.py
         python sentiment_analyzer.py --keyword "iPhone" --posts 200 --no-show
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
LABELS: List[str] = ["Negative", "Neutral", "Positive"]  # alphabetical = sklearn order
COLORS: Dict[str, str] = {
    "Negative": "#e74c3c",
    "Neutral": "#95a5a6",
    "Positive": "#2ecc71",
}

# English stop words, but keep negations: "not good" must differ from "good".
STOP_WORDS: List[str] = sorted(
    set(ENGLISH_STOP_WORDS) - {"not", "no", "nor", "never", "cannot", "nothing"}
)

# --------------------------------------------------------------------------- #
# 1. Dataset simulation
# --------------------------------------------------------------------------- #
SUBJECTS: Dict[str, List[str]] = {
    "app_feedback": [
        "the new update", "the app", "the latest version", "the dark mode",
        "the login flow", "customer support", "the notification system",
        "the camera feature", "the search bar", "the checkout screen",
        "the sync feature", "the new interface",
    ],
    "tech_updates": [
        "the new AI model", "this firmware update", "the cloud outage fix",
        "the new chip", "the 5G rollout", "the open source release",
        "the latest OS patch", "the battery tech", "the new laptop line",
        "the API change", "the new browser build", "the smart home hub",
    ],
    "public_opinion": [
        "the new metro line", "the city internet policy", "the data privacy law",
        "the remote work policy", "the online education reform",
        "the digital ID program", "the public wifi rollout",
        "the streaming price hike", "the new traffic rules",
        "the tech regulation bill", "the social media ban proposal",
        "the e-scooter program",
    ],
}

OPENERS: Dict[str, List[str]] = {
    "Positive": ["Honestly,", "Wow,", "Just tried it:", "Love this:", "Big thumbs up,", ""],
    "Negative": ["Ugh,", "Seriously,", "Not impressed:", "Day 3 and", "Sad to say,", ""],
    "Neutral": ["FYI,", "Update:", "News:", "Reminder:", "Reading that", ""],
}

PHRASES: Dict[str, List[str]] = {
    "Positive": [
        "is absolutely fantastic", "works flawlessly now", "is so much better than before",
        "exceeded my expectations", "is a total game changer", "feels fast and smooth",
        "is brilliant, well done to the team", "makes my life so much easier",
        "is amazing and super intuitive", "is the best improvement this year",
        "is impressively reliable", "got me really excited", "is great value and works well",
        "is excellent, highly recommend it",
    ],
    "Negative": [
        "is terrible", "keeps crashing constantly", "is a complete disaster",
        "is so slow and frustrating", "is a total waste of time", "broke everything again",
        "is awful and unusable", "is the worst thing ever", "is not working and nobody cares",
        "is buggy, confusing and disappointing", "makes me want to quit using it",
        "is overpriced and poorly designed", "is a mess, really angry about it",
        "never works when I need it",
    ],
    "Neutral": [
        "was announced today", "will be released next week", "is available on all platforms",
        "was discussed at the meeting", "has the details listed in the docs",
        "is scheduled for Tuesday", "was mentioned in the quarterly report",
        "is being rolled out in phases", "was covered in this morning's briefing",
        "now supports two more languages", "is one of several options on the table",
        "has a public comment period open", "will be reviewed by the committee",
        "changed the settings menu layout",
    ],
}

# Mixed-signal clauses make the task less trivial (more realistic).
HEDGES: Dict[str, List[str]] = {
    "Positive": [", even if setup took a minute", ", despite the slow start", ", though the docs could be better"],
    "Negative": [", even if the design looks nice", ", despite the hype", ", though the idea was good"],
    "Neutral": [],
}
NEUTRAL_TAILS: List[str] = [", per the release notes", ", as reported this morning", ", according to the blog"]
HASHTAGS: List[str] = ["", "", "", " #tech", " #apps", " #opinion", " #news", " #update"]
ENDINGS: List[str] = [".", "!", "...", ""]


def generate_post(
    sentiment: str, rng: random.Random, subject: Optional[str] = None
) -> Tuple[str, str]:
    """Generate one synthetic post. Returns (text, domain)."""
    domain = rng.choice(list(SUBJECTS))
    subj = subject if subject else rng.choice(SUBJECTS[domain])
    parts = [OPENERS[sentiment] and rng.choice(OPENERS[sentiment]), subj, rng.choice(PHRASES[sentiment])]
    text = " ".join(p for p in parts if p)
    if HEDGES[sentiment] and rng.random() < 0.15:
        text += rng.choice(HEDGES[sentiment])
    if sentiment == "Neutral" and rng.random() < 0.4:
        text += rng.choice(NEUTRAL_TAILS)
    text += rng.choice(ENDINGS) + rng.choice(HASHTAGS)
    return text[0].upper() + text[1:], domain


def build_dataset(n_samples: int = 6000, seed: int = 42, label_noise: float = 0.03) -> pd.DataFrame:
    """Create a balanced, de-duplicated dataset of simulated social posts."""
    rng = random.Random(seed)
    per_class = math.ceil(n_samples / len(LABELS))
    rows = []
    for label in LABELS:
        seen = set()
        attempts = 0
        while len(seen) < per_class and attempts < per_class * 100:
            attempts += 1
            text, domain = generate_post(label, rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append({"text": text, "label": label, "domain": domain})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Label noise (annotator disagreement) so metrics aren't an unrealistic 100%.
    noisy_idx = df.sample(frac=label_noise, random_state=seed).index
    for i in noisy_idx:
        df.at[i, "label"] = rng.choice([l for l in LABELS if l != df.at[i, "label"]])
    return df


# --------------------------------------------------------------------------- #
# 2. Text cleaning
# --------------------------------------------------------------------------- #
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/@mentions/punctuation, collapse whitespace.
    (Stop-word removal is done by TfidfVectorizer.)"""
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)  # also turns '#tag' into 'tag'
    return _SPACES.sub(" ", text).strip()


# --------------------------------------------------------------------------- #
# 3. Model
# --------------------------------------------------------------------------- #
@dataclass
class SentimentModel:
    vectorizer: TfidfVectorizer
    classifier: MultinomialNB
    metrics: Dict

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        cleaned = [clean_text(t) for t in texts]
        features = self.vectorizer.transform(cleaned)
        probs = self.classifier.predict_proba(features)
        classes = self.classifier.classes_
        results = []
        for text, row in zip(texts, probs):
            best = int(np.argmax(row))
            results.append(
                {
                    "text": text,
                    "sentiment": str(classes[best]),
                    "confidence": round(float(row[best]), 4),
                    "probabilities": {
                        str(c).lower(): round(float(p), 4) for c, p in zip(classes, row)
                    },
                }
            )
        return results

    def predict_one(self, text: str) -> Dict:
        return self.predict_batch([text])[0]


def train_model(n_samples: int = 6000, seed: int = 42, verbose: bool = True) -> SentimentModel:
    """Build data, 80/20 split, TF-IDF, train MultinomialNB, report metrics."""
    df = build_dataset(n_samples=n_samples, seed=seed)
    df["clean"] = df["text"].map(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.20, random_state=seed, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = MultinomialNB(alpha=0.5)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    if verbose:
        print("=" * 64)
        print("MODEL TRAINING COMPLETE  (TF-IDF + MultinomialNB)")
        print("=" * 64)
        print(f"Dataset size   : {len(df)} posts  ({df['label'].value_counts().to_dict()})")
        print(f"Train / Test   : {len(X_train)} / {len(X_test)}  (80% / 20%)")
        print(f"Vocabulary     : {len(vectorizer.vocabulary_)} features")
        print(f"Test accuracy  : {accuracy:.4f}\n")
        print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred; order: " + ", ".join(LABELS) + ")")
        print(cm)
        print("=" * 64)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "report": report,
        "confusion_matrix": cm.tolist(),
    }
    return SentimentModel(vectorizer=vectorizer, classifier=clf, metrics=metrics)


# --------------------------------------------------------------------------- #
# 4. Simulated live stream (drop-in stand-in for the Twitter/X API)
# --------------------------------------------------------------------------- #
def _keyword_weights(keyword: str) -> List[float]:
    """Deterministic sentiment mix per keyword (so repeated calls are consistent)."""
    digest = hashlib.md5(keyword.lower().encode("utf-8")).digest()
    raw = [1.0 + digest[i] / 128.0 for i in range(3)]  # 1.0 .. ~3.0
    total = sum(raw)
    return [r / total for r in raw]


def simulate_live_posts(keyword: str, limit: int = 50, window: int = 0) -> List[Dict]:
    """Return `limit` fake recent posts mentioning `keyword`.
    `window` shifts the random stream so successive windows show a trend."""
    base = _keyword_weights(keyword)
    seed = int(hashlib.md5(f"{keyword.lower()}|{window}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    weights = [w * rng.uniform(0.6, 1.4) for w in base]  # per-window drift
    now = datetime.now(timezone.utc)
    posts = []
    for i in range(limit):
        label = rng.choices(LABELS, weights=weights, k=1)[0]
        text, _ = generate_post(label, rng, subject=keyword)
        posts.append(
            {
                "id": f"sim-{window}-{i}",
                "text": text,
                "created_at": (now - timedelta(minutes=rng.randint(0, 60 * (window + 1)))).isoformat(),
            }
        )
    return posts


# --------------------------------------------------------------------------- #
# 5. Visualization
# --------------------------------------------------------------------------- #
def summarize(predictions: List[Dict]) -> Dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for p in predictions:
        counts[p["sentiment"]] += 1
    return counts


def plot_trends(model: SentimentModel, keyword: str, posts_per_window: int,
                windows: int, output: str, show: bool) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Per-window predictions
    window_counts = []
    for w in range(windows):
        preds = model.predict_batch([p["text"] for p in simulate_live_posts(keyword, posts_per_window, w)])
        window_counts.append(summarize(preds))

    latest = window_counts[-1]
    total_latest = sum(latest.values())

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Public Opinion Sentiment Trends - '{keyword}'", fontsize=16, fontweight="bold")

    # --- Bar chart (latest window) ---
    values = [latest[l] for l in LABELS]
    bars = ax_bar.bar(LABELS, values, color=[COLORS[l] for l in LABELS], edgecolor="black", width=0.6)
    ax_bar.set_axisbelow(True)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.6)
    ax_bar.set_ylabel("Number of posts")
    ax_bar.set_title(f"Latest window ({total_latest} posts)")
    ax_bar.set_ylim(0, max(values) * 1.18 if max(values) else 1)
    for bar, v in zip(bars, values):
        pct = 100 * v / total_latest if total_latest else 0
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                    f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # --- Line chart (trend over windows) ---
    xs = list(range(1, windows + 1))
    for label in LABELS:
        ys = [100 * c[label] / sum(c.values()) for c in window_counts]
        ax_line.plot(xs, ys, marker="o", linewidth=2.2, color=COLORS[label], label=label)
        for x, y in zip(xs, ys):
            ax_line.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                             ha="center", fontsize=9, color=COLORS[label])
    ax_line.set_axisbelow(True)
    ax_line.grid(True, linestyle="--", alpha=0.6)
    ax_line.set_xticks(xs)
    ax_line.set_xlabel("Time window (oldest -> newest)")
    ax_line.set_ylabel("Share of posts (%)")
    ax_line.set_ylim(0, 100)
    ax_line.set_title("Sentiment share over time")
    ax_line.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=150)
    print(f"Chart saved to: {output}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sentiment model and plot opinion trends.")
    parser.add_argument("--keyword", default="AI", help="Topic/keyword to analyse (default: AI)")
    parser.add_argument("--posts", type=int, default=150, help="Posts per time window (default: 150)")
    parser.add_argument("--windows", type=int, default=6, help="Number of time windows (default: 6)")
    parser.add_argument("--output", default="sentiment_trends.png", help="Output PNG path")
    parser.add_argument("--no-show", action="store_true", help="Save the PNG without opening a window")
    args = parser.parse_args()

    model = train_model(verbose=True)
    plot_trends(model, args.keyword, args.posts, args.windows, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
