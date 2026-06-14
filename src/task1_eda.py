"""
Task 1: Exploratory Data Analysis
NLP Project 1.1: Sentiment Analysis of Financial News

This script loads Sentences_50Agree.txt, computes descriptive statistics,
and saves tables/figures for the report.
"""

from pathlib import Path
from collections import Counter
import re

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Sentences_50Agree.txt"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load Financial PhraseBank data where each line has sentence@label."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Please copy Sentences_50Agree.txt into the data/ folder."
        )

    records = []
    with path.open("r", encoding="ISO-8859-1") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            if "@" not in line:
                print(f"Warning: skipped line {line_number}, no label separator found.")
                continue
            sentence, label = line.rsplit("@", 1)
            records.append({"sentence": sentence.strip(), "label": label.strip()})

    df = pd.DataFrame(records)
    expected = {"positive", "neutral", "negative"}
    observed = set(df["label"].unique())
    if not observed.issubset(expected):
        print(f"Warning: unexpected labels found: {observed - expected}")
    return df


def tokenize_basic(text: str) -> list[str]:
    """Simple tokenizer for EDA: lowercase and keep alphabetic tokens."""
    return re.findall(r"[A-Za-z]+", text.lower())


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add token and character length features."""
    df = df.copy()
    df["tokens"] = df["sentence"].apply(tokenize_basic)
    df["word_count"] = df["tokens"].apply(len)
    df["char_count"] = df["sentence"].str.len()
    return df


def save_class_distribution(df: pd.DataFrame) -> None:
    """Save class counts and class distribution figure."""
    counts = df["label"].value_counts().rename_axis("label").reset_index(name="count")
    counts["percentage"] = (counts["count"] / len(df) * 100).round(2)
    counts.to_csv(RESULTS_DIR / "class_distribution.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(counts["label"], counts["count"])
    plt.title("Sentiment Class Distribution")
    plt.xlabel("Sentiment label")
    plt.ylabel("Number of sentences")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=300)
    plt.close()


def save_length_statistics(df: pd.DataFrame) -> None:
    """Save text length statistics and distribution plot."""
    stats = df.groupby("label")[["word_count", "char_count"]].describe().round(2)
    stats.to_csv(RESULTS_DIR / "text_length_statistics.csv")

    plt.figure(figsize=(7, 4))
    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label]
        plt.hist(subset["word_count"], bins=25, alpha=0.5, label=label)
    plt.title("Text Length Distribution by Sentiment")
    plt.xlabel("Number of words")
    plt.ylabel("Number of sentences")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "text_length_distribution.png", dpi=300)
    plt.close()


def get_top_ngrams(texts: pd.Series, ngram_range=(1, 1), top_k=20) -> pd.DataFrame:
    """Return most frequent n-grams after simple stop-word filtering."""
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=list(ENGLISH_STOP_WORDS),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        ngram_range=ngram_range,
    )
    matrix = vectorizer.fit_transform(texts)
    counts = matrix.sum(axis=0).A1
    features = vectorizer.get_feature_names_out()
    ngrams = pd.DataFrame({"ngram": features, "count": counts})
    return ngrams.sort_values("count", ascending=False).head(top_k)


def save_ngram_analysis(df: pd.DataFrame) -> None:
    """Save frequent unigram and bigram tables/figures."""
    unigrams = get_top_ngrams(df["sentence"], (1, 1), 20)
    bigrams = get_top_ngrams(df["sentence"], (2, 2), 20)
    unigrams.to_csv(RESULTS_DIR / "top_unigrams.csv", index=False)
    bigrams.to_csv(RESULTS_DIR / "top_bigrams.csv", index=False)

    for table, name, title in [
        (unigrams, "top_unigrams.png", "Top 20 Unigrams"),
        (bigrams, "top_bigrams.png", "Top 20 Bigrams"),
    ]:
        plt.figure(figsize=(8, 5))
        ordered = table.sort_values("count")
        plt.barh(ordered["ngram"], ordered["count"])
        plt.title(title)
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / name, dpi=300)
        plt.close()


def save_label_specific_words(df: pd.DataFrame) -> None:
    """Save frequent words per class for qualitative interpretation."""
    rows = []
    stop_words = set(ENGLISH_STOP_WORDS)
    for label in sorted(df["label"].unique()):
        all_tokens = []
        for tokens in df.loc[df["label"] == label, "tokens"]:
            all_tokens.extend([tok for tok in tokens if tok not in stop_words and len(tok) > 1])
        for word, count in Counter(all_tokens).most_common(20):
            rows.append({"label": label, "word": word, "count": count})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "top_words_by_label.csv", index=False)


def main() -> None:
    df = load_dataset(DATA_PATH)
    df = add_length_features(df)

    df.to_csv(RESULTS_DIR / "loaded_dataset_with_lengths.csv", index=False)
    save_class_distribution(df)
    save_length_statistics(df)
    save_ngram_analysis(df)
    save_label_specific_words(df)

    print("Task 1 completed.")
    print(f"Number of instances: {len(df)}")
    print(f"Labels: {sorted(df['label'].unique())}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
