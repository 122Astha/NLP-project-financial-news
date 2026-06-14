# Short Version for ACM Report: Task 4

For Task 4, 15 positive financial news sentences were selected and represented by averaged word vectors. Each sentence vector was created by averaging the vectors of the words in the sentence. Pairwise cosine similarity was then calculated manually using the dot product and vector norms, without using a pre-built similarity function.

The most similar pairs usually contained related financial topics, such as growth, profit, sales, or positive company performance. The least similar pairs were still positive in sentiment, but they discussed different events or business contexts. This shows that semantic similarity is different from sentiment polarity: two sentences can both be positive while still being semantically different.

The method is simple and interpretable, but it has limitations. Averaging word vectors ignores word order and does not model phrase-level meaning. This can be problematic in financial text, where short expressions may carry important meaning. Therefore, averaged word vectors are useful as a transparent baseline, but they should not be treated as a complete representation of sentence meaning.
