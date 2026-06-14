# Task 4: Textual Similarity - Auto-Filled Draft

For Task 4, 15 positive financial news sentences were selected from the dataset. Each sentence was represented by averaging the vectors of the words in the sentence. In this run, the vector source was: **reproducible dataset-based co-occurrence vectors (fallback; no spaCy vector model found)**.

Cosine similarity was implemented manually. The calculation uses the dot product of two sentence vectors divided by the product of their vector lengths. No pre-built cosine similarity function was used.

## Selected positive sentences

- S01: Finnish pharmaceuticals company Orion 's net sales rose to EUR 190mn in the first quarter of 2009 from EUR 180mn in the first quarter of 2008 .
- S02: The loss for the third quarter of 2007 was EUR 0.3 mn smaller than the loss of the second quarter of 2007 .
- S03: This new deal has strengthened the partnership from Telemig Celular and Tecnomen , which it has started since the beginning of Telemig 's prepaid operations .
- S04: Alma Media expects its net sales to increase as forecast previously .
- S05: Finnish mobile operator DNA will function as a subcontractor to Maingate and will be responsible for telecommunications connections .
- S06: Self-service and automation are in a bigger role now and Fujitsu 's global resources will be exploited effectively .
- S07: The increase in capital stock has been registered in the Finnish Trade Register on 20 November 2006 .
- S08: Adjusted for changes in the Group structure , the Division 's net sales increased by 1.7 % .
- S09: This will bring cost savings of about EUR 3mn a year .
- S10: W+ñrtsil+ñ 's solution has been selected for its low fuel consumption , environmentally sound technology , and global service support .
- S11: The adapter , awarded with the `` Certified Integration for SAP -« ; NetWeaver '' endorsement , integrates Basware s invoice automation and procurement solutions with more than 200 different ERP systems .
- S12: Operating profit was EUR 9.8 mn , compared to a loss of EUR 12.7 mn in the corresponding period in 2009 .
- S13: Svyturys-Utenos Alus , which is controlled by the Nordic group Baltic Beverages Holding ( BBH ) , posted a 4.7-per-cent growth in beer sales for January-May to 46.22 million litres .
- S14: Finnish steel maker Rautaruukki Oyj ( Ruukki ) said on July 7 , 2008 that it won a 9.0 mln euro ( $ 14.1 mln ) contract to supply and install steel superstructures for Partihallsforbindelsen bridge project in Gothenburg , western Sweden .
- S15: Poyry has a good track record of major transportation projects in Latin America .

## Result interpretation

The most similar pair was **S01** and **S02**, with a cosine similarity of **0.9765**.

- S01: Finnish pharmaceuticals company Orion 's net sales rose to EUR 190mn in the first quarter of 2009 from EUR 180mn in the first quarter of 2008 .
- S02: The loss for the third quarter of 2007 was EUR 0.3 mn smaller than the loss of the second quarter of 2007 .

This pair is close in the vector space because both sentences contain related financial vocabulary and describe positive business development. The high similarity does not only mean that both are positive; it also suggests that the words used in the two sentences are semantically related.

The least similar pair was **S11** and **S12**, with a cosine similarity of **0.7790**.

- S11: The adapter , awarded with the `` Certified Integration for SAP -« ; NetWeaver '' endorsement , integrates Basware s invoice automation and procurement solutions with more than 200 different ERP systems .
- S12: Operating profit was EUR 9.8 mn , compared to a loss of EUR 12.7 mn in the corresponding period in 2009 .

This pair is still positive in sentiment, but the sentences discuss different business situations. This shows that sentiment similarity and semantic similarity are not identical. Two sentences can have the same positive label while still being far apart in meaning.

Averaging word vectors is simple and interpretable, but it has limitations. It ignores word order and treats words with similar importance. In financial language, phrase-level meaning can be important, so this method should be understood as a transparent baseline rather than a complete semantic model.
