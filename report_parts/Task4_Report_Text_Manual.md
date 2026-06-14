# Task 4: Textual Similarity - Report Section

For the textual similarity task, 15 sentences with positive sentiment were selected from the dataset. The purpose of this task was not to classify the sentences, but to examine how close positive financial news sentences are to each other in meaning. Each sentence was converted into a sentence-level vector by averaging the word vectors of the words contained in that sentence.

The similarity between two sentence vectors was calculated with cosine similarity. I implemented the cosine similarity calculation manually by computing the dot product of two vectors and dividing it by the product of their vector lengths. This made the calculation transparent and avoided relying on a pre-built similarity function.

The most similar sentence pairs usually had overlapping financial meaning, for example sentences about improved sales, increased profit, growth, contracts, or positive business performance. These pairs received higher cosine similarity values because their averaged word vectors contained related vocabulary and similar semantic information.

The least similar pairs were still positive in sentiment, but they often referred to different business topics. For example, one sentence may describe a company agreement, while another may describe earnings growth or market expansion. This shows that sentiment similarity and semantic similarity are not the same thing. Two sentences can both be positive, but still be semantically far apart because they discuss different events.

A limitation of this method is that averaging word vectors ignores word order and gives each word similar importance. In financial language, small phrases can be important, and the meaning can change depending on context. For example, the word “profit” can appear in both positive and negative settings depending on the surrounding words. Therefore, averaged word vectors are useful as a simple and interpretable baseline, but they cannot capture all details of financial sentence meaning.
