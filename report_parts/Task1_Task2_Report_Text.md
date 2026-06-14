# Report Text for Task 1 and Task 2

This text is written so it can be adapted into the ACM report. Replace the bracketed values after running the scripts and checking the generated tables/figures.

## Task 1: Exploratory Data Analysis

Before training classification models, we performed an exploratory analysis of the Financial PhraseBank dataset using the file `Sentences_50Agree.txt`. The dataset contains short financial news sentences annotated with three sentiment labels: positive, neutral, and negative. Each line was loaded by splitting only at the final `@` character, because the sentence itself may contain special characters. Empty lines were ignored and the original sentence text was preserved for later comparison with preprocessed variants.

The class distribution shows that the dataset is not perfectly balanced. In particular, the neutral class is expected to be frequent in financial news because many headlines describe company events, market updates, or factual announcements without clear positive or negative evaluation. This class imbalance is important for the classification task because a model may obtain reasonable accuracy by predicting the majority class too often. Therefore, macro F1, precision, recall, and confusion matrices should be used in later tasks instead of relying only on accuracy.

We also analyzed sentence length using both word counts and character counts. Since the dataset consists of short financial news headlines or sentences, most texts contain only a small number of tokens. This makes the task challenging because sentiment must often be inferred from very few words. Short texts may lack broader context, for example whether a revenue change is good or bad compared with expectations. As a result, some errors are likely to occur for sentences that use factual financial language without explicit sentiment words.

The vocabulary and n-gram analysis helps identify frequent domain-specific terms. Common words are expected to include financial and company-related expressions such as revenue, profit, sales, shares, company, market, and quarter. These words may appear across multiple sentiment classes, meaning that they are not always sentiment indicators by themselves. For example, the word “profit” can occur in both positive and negative contexts depending on whether profit increased or decreased. Therefore, later models should benefit from representations that capture combinations of words, such as bigrams or TF-IDF features, rather than only isolated token counts.

Overall, the EDA suggests three main challenges for sentiment classification. First, class imbalance may bias models toward the neutral class. Second, short sentence length limits contextual information. Third, financial vocabulary can be ambiguous because the same terms may express different sentiment depending on surrounding words such as “increase”, “fall”, “loss”, or “expectation”. These observations motivate the use of careful evaluation metrics and error analysis in later modeling tasks.

Suggested figures/tables to include:

- Figure: `figures/class_distribution.png`
- Figure: `figures/text_length_distribution.png`
- Figure: `figures/top_unigrams.png` or `figures/top_bigrams.png`
- Table: class distribution from `results/class_distribution.csv`
- Table: length statistics from `results/text_length_statistics.csv`

## Task 2: Text Preprocessing

For preprocessing, we used a moderate and reproducible pipeline. The goal was not to remove as much information as possible, but to reduce unnecessary variation while preserving signals that are important in financial sentiment analysis. The preprocessing steps were implemented in a separate script and the resulting dataset was saved with both the original text and the processed version.

First, all text was lowercased. This reduces vocabulary sparsity by treating words such as “Profit” and “profit” as the same token. Second, the text was tokenized using a simple regular-expression-based tokenizer. This is sufficient for the short English financial sentences in the dataset and keeps the preprocessing transparent and reproducible. Third, most punctuation was removed because punctuation is not expected to be a major sentiment signal in this dataset.

Numbers were handled carefully. Financial news often contains values such as percentages, revenue amounts, and changes in earnings. Removing numbers completely could remove useful information. However, keeping every exact number may create many rare tokens. Therefore, numeric expressions were replaced with a general `<NUM>` token. This preserves the information that a number occurred while avoiding unnecessary vocabulary growth.

Stop-word removal was applied selectively. General English stop words were removed to reduce noise in bag-of-words and TF-IDF representations. However, negation words such as “no”, “not”, “never”, and “without” were kept because they can directly change sentiment polarity. For example, “not profitable” has a very different meaning from “profitable”. Removing negation would therefore be risky for sentiment analysis.

We did not apply stemming or lemmatization as the default preprocessing choice. Although these methods can reduce vocabulary size, they may also remove useful distinctions in a domain-specific financial dataset. Since the dataset contains short sentences, aggressive normalization could make some examples less interpretable. For this reason, stemming or lemmatization can be tested later as an additional experimental variant, but it is not included in the main preprocessing pipeline.

The final preprocessed dataset is saved as `results/preprocessed_dataset.csv`. It contains the original sentence, the sentiment label, lowercased text, normalized text, and the final preprocessed text. This makes the preprocessing transparent and allows the team to compare model performance using raw text versus preprocessed text.

Suggested table to include:

- Table: `results/preprocessing_decisions.csv`
- Examples: `results/preprocessing_examples.csv`

## Short Critical Reflection

The preprocessing choices are expected to help classical models by reducing vocabulary sparsity and removing frequent non-informative words. However, preprocessing can also remove useful context. For this reason, the later classification experiments should compare raw text and preprocessed text. If preprocessing improves Naive Bayes but not the neural network or transformer models, this would suggest that the effect of preprocessing depends on the model type and representation. This comparison is important because the assignment focuses on understanding methodological decisions rather than only achieving the highest score.
