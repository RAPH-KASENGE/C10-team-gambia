# Agricultural Extension RAG — Smart Retrieval for Farmers

**Team Gambia** | TRI AI Saturdays Cohort 10

A retrieval engine for a Retrieval-Augmented Generation (RAG) system that ranks agricultural extension documents against smallholder farmers' natural-language questions, scored by nDCG@5.

## Dataset

Sourced from the Kaggle competition *Agricultural Extension RAG: Smart Retrieval for Farmers* (TRI AI). The corpus contains **695 short agricultural factsheets**, drawn from FAO, CGIAR, ICRISAT, Plantwise, IITA, AGRA, and national extension services, covering 12 crops (maize, rice, cassava, tomato, common bean, cowpea, groundnut, sorghum, plantain, yam, cocoa, pearl millet) plus general-topic documents (soil, climate, drought). Most documents (637/695) are synthetic (CC0); the rest are LLM-grounded rewrites of CC-BY sources.

Query/label files: `train_queries.csv` (308 farmer questions), `qrels_train.csv` (4,194 graded relevance judgments, scale 0–3: 3=perfect match, 2=relevant, 1=marginal/wrong-crop-same-issue, 0=hard negative), and `test_queries.csv` (200 unseen questions scored by the organizers). The dataset deliberately includes **hard negatives** — documents that share vocabulary with a query but answer a different, adjacent problem (e.g., potassium vs. nitrogen deficiency) — to penalize purely lexical retrieval.

## Training Pipeline

Built iteratively, validating each change against real held-out relevance judgments before keeping it:

1. **Preprocessing.** Documents combine title + body text. Two parallel text variants are built: one for lexical matching (Porter-stemmed, title repeated for a deliberate term-frequency boost) and one for dense embeddings (unstemmed, since transformer subword tokenizers already generalize across morphological variants).
2. **Lexical retrieval.** TF-IDF and BM25 (`rank_bm25`), with `k1`/`b` grid-searched on the stemmed corpus.
3. **Dense retrieval.** `BAAI/bge-small-en-v1.5` sentence embeddings with the model's recommended query instruction prefix.
4. **Fusion.** Convex (min-max normalized, alpha-weighted) combination of BM25 and dense scores, alpha swept from 0.0–0.8. Chosen over Reciprocal Rank Fusion, which was measured to actively degrade the dense signal by discarding score magnitude.
5. **Reranking.** `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks the top candidates from fusion.
6. **Query expansion.** A small dictionary maps regional/colloquial agricultural terms (e.g., "matooke wilt") to the corpus's scientific vocabulary, applied to the lexical retrieval path only.

**Key design decisions, each backed by a measured A/B test on `qrels_train.csv`:**
- Stemming closed a large lexical gap (TF-IDF: 0.494→0.582 nDCG@5) — traced to template document clusters (e.g., "Preventing X" vs. "Managing X") whose sole discriminating word wasn't matching an unstemmed query.
- A larger reranker (`bge-reranker-base`, 278M params) underperformed the smaller MiniLM reranker (22M params) zero-shot on this domain, at ~12x the compute — kept the smaller model.
- Query expansion helped lexical retrieval but *hurt* dense retrieval (-0.019 nDCG@5) — applied only to the BM25/TF-IDF input, never the dense query.
- Title-repetition in document text was tested single vs. doubled for dense retrieval and confirmed to help (+0.005 nDCG@5), not assumed.

## Evaluation

Primary metric: **nDCG@5**, computed locally against `qrels_train.csv` using the competition's stated linear-gain formula (`rel_i / log2(i+1)`), matched exactly to the graders' evaluation.

Because iterative tuning against the same 308 training queries risks overfitting the fusion weight and reranker pool size to that specific set, the final configuration is additionally validated with **5-fold cross-validation** (`sklearn.model_selection.KFold`, shuffled, seed=42) — hyperparameters are chosen once on the full training set, then the fixed pipeline is re-measured across 5 held-out splits to report a mean ± standard deviation rather than a single number. Final result: **mean nDCG@5 = 0.782, std = 0.024** across folds, closely matching the full-training-set score (0.782), indicating the reported number is not an artifact of overfitting the tuning process.

A per-query error-analysis pass sorts training queries by nDCG@5 ascending and inspects retrieved-vs-gold documents directly, which is how the stemming issue above was diagnosed, and how a residual failure cluster (near-identical documents differing by phrasing, e.g. "manage X" vs. "outbreak of X" scoring 0.0 vs. 0.39 on functionally identical queries) was flagged as open work.

Note: local nDCG@5 has not always predicted the public leaderboard ranking between pipeline variants; the simplest dense-only configuration outperformed more complex fusion+rerank variants on the leaderboard despite scoring lower locally — flagged as an open train/test distribution gap.

## Reproduction

1. Open the competition on Kaggle and create a **New Notebook** from the Code tab (submission requires execution on Kaggle's platform, not Colab).
2. Confirm the competition data is attached in the notebook's Input panel.
3. Import `scripts/agri_rag_retrieval_v7.ipynb` from this repository (File → Import Notebook).
4. Run all cells top to bottom. The notebook installs its own dependencies (`rank_bm25`, `sentence-transformers`, `nltk`), auto-locates the data under `/kaggle/input/`, runs all ablations and the 5-fold CV, and writes `submission.csv` to `/kaggle/working/` in the required long format (`QueryId,DocumentId`, 5 ranked rows per query).
5. **Save Version → Save & Run All (Commit)**, then submit the committed version's `submission.csv` via the competition's Submit Prediction flow.

No GPU is required; the full pipeline runs on Kaggle's default CPU environment.

## Appendix

**Contributors:** Team Gambia — Bon_Kurei (and teammates, add names here)
**Mentors:** (add mentor name(s) here)
**Cohort:** TRI AI Saturdays Cohort 10
