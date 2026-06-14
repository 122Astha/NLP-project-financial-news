Hi, I added my final part for Task 4 as well.

The package now includes Task 1 EDA, Task 2 preprocessing, Task 4 textual similarity, report text drafts, and a final checking checklist.

For Task 4, the script selects 15 positive examples, creates averaged sentence vectors, and computes pairwise cosine similarity manually. It saves the selected sentences, similarity matrix, most similar pairs, and least similar pairs in the results folder.

To run everything, put `Sentences_50Agree.txt` inside the `data` folder and run:

```bash
python run_all_my_parts.py
```

Please check the generated Task 4 similarity results before adding them to the final report, because the report text should match the actual output values.
