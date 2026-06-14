"""
Task 4: Textual Similarity

This script selects 15 positive sentiment examples from Sentences_50Agree.txt,
represents each sentence by averaging word vectors, and computes pairwise
cosine similarities manually.

Run:
    python src/task4_similarity.py

Outputs:
    results/task4_selected_positive_sentences.csv
    results/task4_similarity_matrix.csv
    results/task4_most_similar_pairs.csv
    results/task4_least_similar_pairs.csv
    report_parts/Task4_Report_Text_AutoFilled.md
"""

from pathlib import Path
import re
import math
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Sentences_50Agree.txt"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report_parts"
RESULTS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
N_EXAMPLES = 15
VECTOR_DIM_FALLBACK = 50


def load_dataset(path: Path):
    """Load Financial PhraseBank format: sentence@label."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPlease place Sentences_50Agree.txt inside the data/ folder."
        )
    rows = []
    with open(path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            line = line.strip()
            if not line or "@" not in line:
                continue
            sentence, label = line.rsplit("@", 1)
            rows.append({"sentence": sentence.strip(), "label": label.strip().lower()})
    return pd.DataFrame(rows)


def basic_tokenize(text: str):
    """Simple tokenizer for financial news text."""
    text = text.lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", text)


def select_positive_examples(df: pd.DataFrame, n: int = 15):
    """Select 15 positive examples reproducibly."""
    positives = df[df["label"] == "positive"].copy()
    positives["token_count"] = positives["sentence"].apply(lambda x: len(basic_tokenize(x)))
    candidates = positives[(positives["token_count"] >= 5) & (positives["token_count"] <= 35)]
    if len(candidates) < n:
        candidates = positives
    selected = candidates.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    selected.insert(0, "id", [f"S{i+1:02d}" for i in range(len(selected))])
    return selected[["id", "sentence", "label", "token_count"]]


def try_spacy_vectors():
    """Try to load installed spaCy vectors. Optional."""
    try:
        import spacy
    except Exception:
        return None, None
    for model_name in ["en_core_web_md", "en_core_web_lg"]:
        try:
            nlp = spacy.load(model_name)
            if nlp.vocab.vectors_length > 0:
                return nlp, f"spaCy word vectors ({model_name})"
        except Exception:
            pass
    return None, None


def sentence_vector_spacy(sentence: str, nlp):
    vectors = []
    for token in nlp(sentence):
        if token.is_space or token.is_punct:
            continue
        if token.has_vector:
            vectors.append(token.vector.astype(float))
    if not vectors:
        return np.zeros(nlp.vocab.vectors_length, dtype=float)
    return np.mean(np.vstack(vectors), axis=0)


def build_fallback_word_vectors(all_sentences, dim=50):
    """
    Build deterministic dataset-based co-occurrence vectors.
    This fallback keeps the script reproducible if no external word-vector
    model is installed. It is not a replacement for real pre-trained vectors,
    but it allows the task pipeline and manual cosine computation to run.
    """
    random.seed(RANDOM_SEED)
    tokenized = [basic_tokenize(s) for s in all_sentences]
    vocab_counter = Counter(token for sent in tokenized for token in sent)
    vocab = [w for w, c in vocab_counter.items() if c >= 2]

    rng = np.random.default_rng(RANDOM_SEED)
    index_vectors = {}
    for word in vocab:
        vec = rng.normal(0, 1, dim)
        norm = np.linalg.norm(vec)
        index_vectors[word] = vec / norm if norm > 0 else vec

    word_vectors = defaultdict(lambda: np.zeros(dim, dtype=float))
    window = 2
    for sent in tokenized:
        for i, target in enumerate(sent):
            if target not in index_vectors:
                continue
            left = max(0, i - window)
            right = min(len(sent), i + window + 1)
            for j in range(left, right):
                if j == i:
                    continue
                ctx = sent[j]
                if ctx in index_vectors:
                    word_vectors[target] += index_vectors[ctx]

    normalized = {}
    for word, vec in word_vectors.items():
        norm = np.linalg.norm(vec)
        normalized[word] = vec / norm if norm > 0 else vec
    return normalized, dim


def sentence_vector_fallback(sentence: str, word_vectors, dim: int):
    vectors = [word_vectors[t] for t in basic_tokenize(sentence) if t in word_vectors]
    if not vectors:
        return np.zeros(dim, dtype=float)
    return np.mean(np.vstack(vectors), axis=0)


def manual_cosine_similarity(vec_a, vec_b):
    """Manual cosine similarity: dot(a,b) / (||a|| * ||b||)."""
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot_product += float(a) * float(b)
        norm_a += float(a) ** 2
        norm_b += float(b) ** 2
    norm_a = math.sqrt(norm_a)
    norm_b = math.sqrt(norm_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def compute_similarity_matrix(ids, vectors):
    matrix = pd.DataFrame(index=ids, columns=ids, dtype=float)
    for i, id_i in enumerate(ids):
        for j, id_j in enumerate(ids):
            matrix.loc[id_i, id_j] = round(manual_cosine_similarity(vectors[i], vectors[j]), 4)
    return matrix


def pair_table(selected, matrix):
    sentence_lookup = dict(zip(selected["id"], selected["sentence"]))
    ids = list(selected["id"])
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            rows.append({
                "sentence_id_1": id_a,
                "sentence_id_2": id_b,
                "similarity": float(matrix.loc[id_a, id_b]),
                "sentence_1": sentence_lookup[id_a],
                "sentence_2": sentence_lookup[id_b],
            })
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)


def make_report_text(selected, top_pairs, low_pairs, vector_source):
    selected_lines = "\n".join([f"- {r.id}: {r.sentence}" for r in selected.itertuples()])
    top = top_pairs.iloc[0]
    low = low_pairs.iloc[0]
    return f"""# Task 4: Textual Similarity - Auto-Filled Draft

For Task 4, 15 positive financial news sentences were selected from the dataset. Each sentence was represented by averaging the vectors of the words in the sentence. In this run, the vector source was: **{vector_source}**.

Cosine similarity was implemented manually. The calculation uses the dot product of two sentence vectors divided by the product of their vector lengths. No pre-built cosine similarity function was used.

## Selected positive sentences

{selected_lines}

## Result interpretation

The most similar pair was **{top['sentence_id_1']}** and **{top['sentence_id_2']}**, with a cosine similarity of **{top['similarity']:.4f}**.

- {top['sentence_id_1']}: {top['sentence_1']}
- {top['sentence_id_2']}: {top['sentence_2']}

This pair is close in the vector space because both sentences contain related financial vocabulary and describe positive business development. The high similarity does not only mean that both are positive; it also suggests that the words used in the two sentences are semantically related.

The least similar pair was **{low['sentence_id_1']}** and **{low['sentence_id_2']}**, with a cosine similarity of **{low['similarity']:.4f}**.

- {low['sentence_id_1']}: {low['sentence_1']}
- {low['sentence_id_2']}: {low['sentence_2']}

This pair is still positive in sentiment, but the sentences discuss different business situations. This shows that sentiment similarity and semantic similarity are not identical. Two sentences can have the same positive label while still being far apart in meaning.

Averaging word vectors is simple and interpretable, but it has limitations. It ignores word order and treats words with similar importance. In financial language, phrase-level meaning can be important, so this method should be understood as a transparent baseline rather than a complete semantic model.
"""


def main():
    df = load_dataset(DATA_PATH)
    selected = select_positive_examples(df, N_EXAMPLES)
    selected.to_csv(RESULTS_DIR / "task4_selected_positive_sentences.csv", index=False)

    nlp, source = try_spacy_vectors()
    vectors = []
    if nlp is not None:
        vector_source = source
        vectors = [sentence_vector_spacy(s, nlp) for s in selected["sentence"]]
    else:
        word_vectors, dim = build_fallback_word_vectors(df["sentence"].tolist(), VECTOR_DIM_FALLBACK)
        vector_source = "reproducible dataset-based co-occurrence vectors (fallback; no spaCy vector model found)"
        vectors = [sentence_vector_fallback(s, word_vectors, dim) for s in selected["sentence"]]

    matrix = compute_similarity_matrix(selected["id"].tolist(), vectors)
    matrix.to_csv(RESULTS_DIR / "task4_similarity_matrix.csv")

    pairs = pair_table(selected, matrix)
    most_similar = pairs.head(10)
    least_similar = pairs.tail(10).sort_values("similarity", ascending=True)
    most_similar.to_csv(RESULTS_DIR / "task4_most_similar_pairs.csv", index=False)
    least_similar.to_csv(RESULTS_DIR / "task4_least_similar_pairs.csv", index=False)

    report = make_report_text(selected, most_similar, least_similar, vector_source)
    (REPORT_DIR / "Task4_Report_Text_AutoFilled.md").write_text(report, encoding="utf-8")

    print("Task 4 completed successfully.")
    print(f"Vector source: {vector_source}")
    print("Outputs saved in results/ and report_parts/.")


if __name__ == "__main__":
    main()
