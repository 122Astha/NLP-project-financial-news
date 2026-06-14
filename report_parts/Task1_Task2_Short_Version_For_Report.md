# Short Version for Final ACM Report

## Task 1: Exploratory Data Analysis

We first analyzed the `Sentences_50Agree.txt` file to understand the structure of the Financial PhraseBank dataset before training any models. The dataset contains short financial news sentences labeled as positive, neutral, or negative. The class distribution is important because financial news often contains many neutral factual statements, which can create class imbalance. This may bias models toward the majority class and makes macro F1 and class-wise precision/recall more informative than accuracy alone.

We also examined text length using word and character counts. Most sentences are short, which makes classification difficult because there is limited context. In many cases, the sentiment depends on a small number of financial expressions such as “increase”, “decline”, “profit”, “loss”, or “sales”. Frequent word and bigram analysis showed that many financial terms can occur across different classes, so single words are not always reliable sentiment indicators. These observations suggest that later experiments should compare different feature representations and include careful error analysis.

## Task 2: Text Preprocessing

We used a moderate preprocessing pipeline to reduce noise while preserving important financial sentiment information. The text was lowercased, tokenized, and punctuation was removed. Numeric values were replaced with a general `<NUM>` token because numbers are common in financial news and may carry useful information, but exact values would create many rare features.

Stop-word removal was applied carefully. General stop words were removed, but negation words such as “not”, “no”, “never”, and “without” were kept because they can change sentiment polarity. Stemming and lemmatization were not applied in the main pipeline because aggressive normalization may remove useful domain-specific distinctions in short financial sentences. The final preprocessed file keeps both original and processed text, allowing later models to compare raw and preprocessed representations.
