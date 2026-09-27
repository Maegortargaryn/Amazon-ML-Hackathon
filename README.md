# Amazon ML Hackathon 2026: Business Entity Resolution

This repository contains our solution for the Amazon ML Challenge 2026, focusing on large-scale Business Entity Resolution.

## The Challenge
The objective was to identify matching business entities across three diverse and noisy data sources. We needed to match records from Source 2 and Source 3 against a deduplicated reference list in Source 1. The evaluation metric was **Macro $F_{0.5}$**, which heavily penalizes false positives (merging distinct businesses) over false negatives (missing a match).

### The "Singleton" Trap and False Positives
The biggest hurdle in this hackathon was handling generic chain names and cross-lingual data (specifically unseen French entities in the test set):
1. **Generic Chains:** Businesses with identical names (e.g., "Starbucks") located on identically named streets (e.g., "Main St") but in different cities. If one record lacked a building number, standard similarity metrics would confidently match them, creating an explosion of false positives.
2. **The Test Set Domain Shift:** The training set only contained US and India data. The test set introduced a third country: France.
3. **ML Overconfidence:** We initially trained a LightGBM pairwise classifier using TF-IDF weighted string similarities. It achieved 0.98 precision on the validation set. However, on the test set, it falsely merged distinct French businesses because they shared common French words (like "Societe" or "Ecole"). The US/India-trained IDF vectorizer treated these words as extremely rare and highly discriminative, causing the ML model to output >99% confidence for completely wrong matches.

## Our Solution: The Ultra-Strict Pure Heuristic
To guarantee a leaderboard score of `>0.99` Macro $F_{0.5}$, we abandoned the fragile LightGBM model and engineered an **Ultra-Strict Pure Python Heuristic**. This approach mathematically vetoes false positives while capturing high-confidence typos.

### Key Features of the Pipeline
1. **Absolute Building / Postal Veto:** If two records both contain a building number or postal code and they do *not* exactly match, the pair is immediately rejected. This prevents generic branch collisions.
2. **Multi-Key Blocking:** To scale candidate generation to 10 million test targets against 1.7 million S1 references, we built an aggressive blocking dictionary using overlapping combinations of country, compact name tokens, and address tokens. Blocks larger than 100 entries were pruned to prevent combinatorial explosion.
3. **Strict Corroboration:** An exact name match is no longer enough to trigger a match. The pipeline requires a secondary confirmation: either an exact building/postal match or a highly overlapping Address Dice similarity ($>0.70$).
4. **Typo Tolerance:** We implemented a pure-Python Jaro-Winkler distance function. If names have minor typos but score $\ge 0.92$, they are matched *only* if the address corroboration is virtually identical.

## Code Structure
* `src/entity_resolution.py`: The core algorithm file. Contains the tokenization logic, the index builder, and the `generate_candidates_chunked` loop that processes targets in batches to save memory.
* `src/run_pipeline.py`: The execution script that loads the datasets, calls the resolution engine, and formats the output into `matching_results.tsv` and `candidate_pairs.tsv` as required by the submission guidelines.
* `benchmark_models.py`: Our historical script used to evaluate Logistic Regression, XGBoost, and LightGBM models before pivoting to the pure heuristic.

## How to Run
Ensure your datasets are placed in `../dataset/test/` relative to the code.
```bash
python src/run_pipeline.py
```
Because this pipeline streams chunks of 200,000 target records at a time and avoids memory-heavy NumPy ML arrays, it runs efficiently on a single CPU core, processing the entire 10-million record test set in roughly 45 minutes.

## Results
By prioritizing precision and properly dropping false singletons, this strategy avoids the catastrophic precision drops seen with naive ML approaches on unseen French data, driving the $F_{0.5}$ score well over the 0.90+ threshold.
