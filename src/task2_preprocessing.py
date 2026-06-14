"""
Task 2: Text Preprocessing
NLP Project 1.1: Sentiment Analysis of Financial News

This script creates reproducible preprocessing variants for later modeling.
It intentionally keeps preprocessing moderate because financial sentiment can
be affected by numbers, negation, and domain-specific terms.
"""

from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Sentences_50Agree.txt"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Keep negation words because they can change sentiment polarity.
NEGATION_WORDS = {"no", "not", "nor", "never", "without"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS


def load_dataset(path: Path) -> pd.DataFrame:
    """Load Financial PhraseBank data where each line has sentence@label."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Please copy Sentences_50Agree.txt into the data/ folder."
        )

    records = []
    with path.open("r", encoding="ISO-8859-1") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            sentence, label = line.rsplit("@", 1)
            records.append({"sentence": sentence.strip(), "label": label.strip()})
    return pd.DataFrame(records)


def normalize_text(text: str) -> str:
    """
    Basic normalization:
    - lowercase text
    - replace numbers with <NUM> to preserve numeric information in a general form
    - remove punctuation except angle brackets used in <NUM>
    - collapse repeated whitespace
    """
    text = text.lower()
    text = re.sub(r"\b\d+(?:[.,]\d+)?%?\b", " <NUM> ", text)
    text = re.sub(r"[^a-z<>\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text into alphabetic tokens and <NUM>."""
    return re.findall(r"<NUM>|[a-z]+", text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove general English stop words, but keep negation words."""
    return [tok for tok in tokens if tok not in STOP_WORDS]


def preprocess(text: str) -> str:
    """Return final preprocessed text string."""
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


def main() -> None:
    df = load_dataset(DATA_PATH)
    df["text_lower"] = df["sentence"].str.lower()
    df["text_normalized"] = df["sentence"].apply(normalize_text)
    df["text_preprocessed"] = df["sentence"].apply(preprocess)

    df.to_csv(RESULTS_DIR / "preprocessed_dataset.csv", index=False)
    df[["sentence", "label", "text_preprocessed"]].head(25).to_csv(
        RESULTS_DIR / "preprocessing_examples.csv", index=False
    )

    summary = pd.DataFrame(
        [
            {
                "choice": "Lowercasing",
                "decision": "Applied",
                "reason": "Reduces duplicate vocabulary forms such as Profit/profit.",
            },
            {
                "choice": "Tokenization",
                "decision": "Applied",
                "reason": "Needed for bag-of-words, TF-IDF, and frequency analysis.",
            },
            {
                "choice": "Punctuation removal",
                "decision": "Applied",
                "reason": "Most punctuation is not expected to carry strong sentiment in short financial headlines.",
            },
            {
                "choice": "Number handling",
                "decision": "Applied as <NUM>",
                "reason": "Financial news contains many values; replacing them preserves numeric signal without creating many rare tokens.",
            },
            {
                "choice": "Stop-word removal",
                "decision": "Applied carefully",
                "reason": "Common words are removed, but negation terms such as not/no/never are kept because they affect polarity.",
            },
            {
                "choice": "Stemming or lemmatization",
                "decision": "Not applied by default",
                "reason": "The dataset contains short financial sentences, and aggressive word reduction may remove useful domain-specific distinctions.",
            },
        ]
    )
    summary.to_csv(RESULTS_DIR / "preprocessing_decisions.csv", index=False)

    print("Task 2 completed.")
    print(f"Preprocessed dataset saved to: {RESULTS_DIR / 'preprocessed_dataset.csv'}")


if __name__ == "__main__":
    main()
