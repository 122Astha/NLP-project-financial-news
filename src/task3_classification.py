"""
Task 3: Sentiment Classification

This script trains and evaluates several machine learning models for sentiment classification.
The main steps are:
1. Load the preprocessed dataset from Task 2.
2. Split the data into training and test sets.
3. Convert the text into numerical features using Bag-of-Words and TF-IDF.
4. Train and evaluate Naive Bayes and Neural Network models with both feature types.
5. Perform error analysis on the best model.
6. Create a binary dataset without neutral examples and repeat the training and evaluation.

"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "preprocessed_dataset.csv"
RESULTS_DIR = ROOT / "results"


experiment_results = []

# Print a simple section title so the output is easy to follow.
def print_section(title):
    
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

# Print the main evaluation scores for one model.
def show_results(model_name, true_labels, predicted_labels):
    
    print_section(model_name)
    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
    experiment_results.append(
        {
            "experiment": model_name,
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
        }
    )

    print("Accuracy:", accuracy)
    print("Macro F1:", macro_f1)
    print()
    print("Classification report:")
    print(classification_report(true_labels, predicted_labels))
    print("Confusion matrix:")
    print(confusion_matrix(true_labels, predicted_labels))


# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

# We use the preprocessed text from Task 2 as model input.
# fillna("") fixes rare rows that became empty after preprocessing.
texts = df["text_preprocessed"].fillna("")
labels = df["label"]

print_section("1. Load the data")
print("Dataset loaded successfully.")
print("Total rows:", len(df))
print()
print("Class distribution in the full dataset:")
print(labels.value_counts())


# ---------------------------------------------------------------------------
# 2. Split the data into 80% training and 20% test data
# ---------------------------------------------------------------------------

# stratify=labels keeps the same class balance in train and test data.
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

print_section("2. Split the data")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print()
print("Class distribution in the training set:")
print(y_train.value_counts())
print()
print("Class distribution in the test set:")
print(y_test.value_counts())


# ---------------------------------------------------------------------------
# 3. Convert text into numbers
# ---------------------------------------------------------------------------

# Bag-of-Words counts how often each word appears in each sentence.
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# TF-IDF gives lower weight to words that appear in many sentences.
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print_section("3. Convert text into numbers")
print("Bag-of-Words features:", X_train_bow.shape[1])
print("TF-IDF features:", X_train_tfidf.shape[1])


# ---------------------------------------------------------------------------
# 4. Naive Bayes with Bag-of-Words
# ---------------------------------------------------------------------------

nb_bow_model = MultinomialNB()
nb_bow_model.fit(X_train_bow, y_train)
nb_bow_predictions = nb_bow_model.predict(X_test_bow)

show_results("4. Naive Bayes with Bag-of-Words", y_test, nb_bow_predictions)


# ---------------------------------------------------------------------------
# 5. Naive Bayes with TF-IDF
# ---------------------------------------------------------------------------

nb_tfidf_model = MultinomialNB()
nb_tfidf_model.fit(X_train_tfidf, y_train)
nb_tfidf_predictions = nb_tfidf_model.predict(X_test_tfidf)

show_results("5. Naive Bayes with TF-IDF", y_test, nb_tfidf_predictions)


# ---------------------------------------------------------------------------
# 6. Neural Network with Bag-of-Words
# ---------------------------------------------------------------------------

nn_bow_model = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=300,
    random_state=42,
)
nn_bow_model.fit(X_train_bow, y_train)
nn_bow_predictions = nn_bow_model.predict(X_test_bow)

show_results("6. Neural Network with Bag-of-Words", y_test, nn_bow_predictions)


# ---------------------------------------------------------------------------
# 7. Neural Network with TF-IDF
# ---------------------------------------------------------------------------

nn_tfidf_model = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=300,
    random_state=42,
)
nn_tfidf_model.fit(X_train_tfidf, y_train)
nn_tfidf_predictions = nn_tfidf_model.predict(X_test_tfidf)

show_results("7. Neural Network with TF-IDF", y_test, nn_tfidf_predictions)


# ---------------------------------------------------------------------------
# 8. Error analysis for the best current model
# ---------------------------------------------------------------------------

# We inspect mistakes from Naive Bayes with Bag-of-Words because it has the
# best current macro F1 score.
errors = pd.DataFrame(
    {
        "sentence": df.loc[X_test.index, "sentence"],
        "true_label": y_test,
        "predicted_label": nb_bow_predictions,
    }
)

errors = errors[errors["true_label"] != errors["predicted_label"]]
errors.to_csv(RESULTS_DIR / "task3_errors_nb_bow.csv", index=False)

print_section("8. Error analysis")
print("Number of wrong predictions:", len(errors))
print()
print("First 10 wrong predictions:")
print(errors.head(10))


# ---------------------------------------------------------------------------
# 9. Create a binary dataset without neutral examples
# ---------------------------------------------------------------------------

binary_df = df[df["label"] != "neutral"].copy()

binary_texts = binary_df["text_preprocessed"].fillna("")
binary_labels = binary_df["label"]

X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    binary_texts,
    binary_labels,
    test_size=0.2,
    random_state=42,
    stratify=binary_labels,
)

print_section("9. Create binary dataset")
print("Rows after removing neutral:", len(binary_df))
print("Binary training rows:", len(X_train_binary))
print("Binary test rows:", len(X_test_binary))
print()
print("Binary class distribution:")
print(binary_labels.value_counts())


# ---------------------------------------------------------------------------
# 10. Convert binary text into numbers
# ---------------------------------------------------------------------------

binary_bow_vectorizer = CountVectorizer()
X_train_binary_bow = binary_bow_vectorizer.fit_transform(X_train_binary)
X_test_binary_bow = binary_bow_vectorizer.transform(X_test_binary)

binary_tfidf_vectorizer = TfidfVectorizer()
X_train_binary_tfidf = binary_tfidf_vectorizer.fit_transform(X_train_binary)
X_test_binary_tfidf = binary_tfidf_vectorizer.transform(X_test_binary)

print_section("10. Convert binary text into numbers")
print("Binary Bag-of-Words features:", X_train_binary_bow.shape[1])
print("Binary TF-IDF features:", X_train_binary_tfidf.shape[1])


# ---------------------------------------------------------------------------
# 11. Binary Naive Bayes with Bag-of-Words
# ---------------------------------------------------------------------------

binary_nb_bow_model = MultinomialNB()
binary_nb_bow_model.fit(X_train_binary_bow, y_train_binary)
binary_nb_bow_predictions = binary_nb_bow_model.predict(X_test_binary_bow)

show_results(
    "11. Binary Naive Bayes with Bag-of-Words",
    y_test_binary,
    binary_nb_bow_predictions,
)


# ---------------------------------------------------------------------------
# 12. Binary Naive Bayes with TF-IDF
# ---------------------------------------------------------------------------

binary_nb_tfidf_model = MultinomialNB()
binary_nb_tfidf_model.fit(X_train_binary_tfidf, y_train_binary)
binary_nb_tfidf_predictions = binary_nb_tfidf_model.predict(X_test_binary_tfidf)

show_results(
    "12. Binary Naive Bayes with TF-IDF",
    y_test_binary,
    binary_nb_tfidf_predictions,
)


# ---------------------------------------------------------------------------
# 13. Binary Neural Network with Bag-of-Words
# ---------------------------------------------------------------------------

binary_nn_bow_model = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=300,
    random_state=42,
)
binary_nn_bow_model.fit(X_train_binary_bow, y_train_binary)
binary_nn_bow_predictions = binary_nn_bow_model.predict(X_test_binary_bow)

show_results(
    "13. Binary Neural Network with Bag-of-Words",
    y_test_binary,
    binary_nn_bow_predictions,
)


# ---------------------------------------------------------------------------
# 14. Binary Neural Network with TF-IDF
# ---------------------------------------------------------------------------

binary_nn_tfidf_model = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=300,
    random_state=42,
)
binary_nn_tfidf_model.fit(X_train_binary_tfidf, y_train_binary)
binary_nn_tfidf_predictions = binary_nn_tfidf_model.predict(X_test_binary_tfidf)

show_results(
    "14. Binary Neural Network with TF-IDF",
    y_test_binary,
    binary_nn_tfidf_predictions,
)


# ---------------------------------------------------------------------------
# 15. Save a small summary table 
# ---------------------------------------------------------------------------

summary = pd.DataFrame(experiment_results)
summary.to_csv(RESULTS_DIR / "task3_model_summary.csv", index=False)

print_section("15. Save summary")
print("Saved:", RESULTS_DIR / "task3_model_summary.csv")
