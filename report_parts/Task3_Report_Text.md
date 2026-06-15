# Task 3: Sentiment Classification

For Task 3, we trained classical machine learning models for financial sentiment classification. The dataset was first split into training and test data using an 80/20 split. We used stratification during the split, meaning that the proportions of negative, neutral, and positive examples were kept similar in both parts. This is important because the dataset is imbalanced, with many more neutral examples than negative examples.

The models used the preprocessed text from Task 2. In this version of the data, the text was lowercased, punctuation was removed, numbers were replaced with the token `<num>`, and common stop words were removed while negation words such as "no" and "not" were kept. This preprocessing was chosen to reduce vocabulary size while preserving information that can be important for sentiment.

Before training the models, the text had to be converted into numerical features. We tested two representations. The first representation was Bag-of-Words, which counts how often each word appears in a sentence. The second representation was TF-IDF, which also uses word frequency but gives less weight to words that appear very often across many sentences.

## Multiclass Classification Results

We first performed multiclass classification with all three sentiment labels: negative, neutral, and positive. Four experiments were tested. Naive Bayes with Bag-of-Words achieved an accuracy of 0.7134 and a macro F1 score of 0.6519. Naive Bayes with TF-IDF achieved an accuracy of 0.6794 and a macro F1 score of 0.4351. The feed-forward neural network with Bag-of-Words achieved an accuracy of 0.7000 and a macro F1 score of 0.6428. The feed-forward neural network with TF-IDF achieved an accuracy of 0.6897 and a macro F1 score of 0.6327.

The best overall multiclass result was achieved by Naive Bayes with Bag-of-Words. The neural network with Bag-of-Words had a similar macro F1 score, but it did not clearly improve over the simpler Naive Bayes baseline.

The weakest result was Naive Bayes with TF-IDF. Although its accuracy was 0.6794, its macro F1 score was only 0.4351. This happened because the model almost never predicted the negative class correctly. This result shows why accuracy alone is not enough for this dataset. Since the dataset is imbalanced, a model can obtain reasonable accuracy by predicting the majority neutral class often, while still performing poorly on the smaller negative class.

## Interpretation

The neutral class was generally the easiest class for the models. This is expected because neutral is the largest class in the dataset. The negative class was more difficult because it has far fewer training examples. Positive and neutral sentences were also often confused with each other. One possible reason is that financial news is frequently written in a factual style, so even positive business events can look similar to neutral announcements.

The comparison between Bag-of-Words and TF-IDF shows that a more complex representation is not always better. For Naive Bayes, Bag-of-Words performed much better than TF-IDF. For the neural network, the two representations were closer, but neither neural network clearly outperformed the Naive Bayes baseline. This suggests that the simple word-count model is already a strong baseline for this dataset.

## Error Analysis

For error analysis, we inspected the wrong predictions of the best current model, Naive Bayes with Bag-of-Words. This model made 278 wrong predictions on the test set. The largest error type was positive sentences predicted as neutral. This happened 109 times. A likely reason is that many positive financial sentences are written in a factual style and do not contain strongly emotional words. For example, a sentence about increased share capital or expanded staff can be positive from a business perspective, but it may look like a neutral company announcement to the model.

Another common error type was neutral sentences predicted as positive. This happened 79 times. Many of these sentences contain words such as "profit", "sales", "gains", or financial amounts. These words often appear in positive examples, so the model may treat them as positive signals even when the sentence is only reporting factual information. This shows a limitation of simple word-count models: they do not fully understand whether a financial term is actually positive in context.

Negative examples were also sometimes predicted as neutral or positive. Some negative sentences describe decreases, weak demand, or lower share prices, but the negative meaning depends on the relationship between numbers or on financial context. For example, a sentence about sales declining or demand being weak is negative, but a simple model may focus on company names, market terms, or numerical expressions instead of the negative event.

Overall, the error analysis shows three main problems. First, the class imbalance makes the model more reliable for neutral than for the smaller classes. Second, financial sentiment often depends on context, not only individual words. Third, positive, neutral, and negative sentences can share many of the same financial terms, so simple word-count features are limited.

## Binary Classification

Finally, we repeated the classification task in a binary setting by removing all neutral examples. This left only positive and negative sentences. After removing neutral, the dataset contained 1,967 examples: 1,363 positive and 604 negative. We again used an 80/20 stratified train-test split.

In the binary setting, Naive Bayes with Bag-of-Words achieved an accuracy of 0.8071 and a macro F1 score of 0.7734. Naive Bayes with TF-IDF achieved an accuracy of 0.7589 and a macro F1 score of 0.6069. The feed-forward neural network with Bag-of-Words achieved an accuracy of 0.7995 and a macro F1 score of 0.7660. The feed-forward neural network with TF-IDF achieved an accuracy of 0.7995 and a macro F1 score of 0.7650.

The binary results were clearly better than the multiclass results. The best binary model was again Naive Bayes with Bag-of-Words. The neural network models were close, but they did not clearly outperform the simpler Naive Bayes model.

Removing the neutral class made the task easier because the model no longer had to distinguish factual neutral business news from positive or negative financial news. In the multiclass setting, many mistakes happened between positive and neutral examples. In the binary setting, this source of confusion was removed. However, the binary task was still not perfect because positive examples were still more frequent than negative examples, and some financial sentences require context to interpret correctly.

The TF-IDF version of Naive Bayes again performed worse than Bag-of-Words. In the binary setting, it predicted the positive class very often and missed many negative examples. This confirms the earlier observation that TF-IDF was not the best representation for Naive Bayes on this dataset.

## Current Conclusion

For both the multiclass and binary settings, Naive Bayes with Bag-of-Words was the best overall model. It is simple, fast, and performed at least as well as the feed-forward neural network. The results also show that macro F1 is especially important for evaluation because it reveals weaknesses on minority classes, especially the negative class.

The binary classification results were stronger than the multiclass results, which shows that the neutral class adds substantial difficulty. This is expected because many neutral financial sentences contain terms such as profit, sales, or gains, which can also appear in positive or negative examples. Overall, the experiments suggest that simple classical models can provide strong baselines, but they still struggle with financial context and class imbalance.
