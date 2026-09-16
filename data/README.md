# Data

This project uses the dataset from the Kaggle competition
**"Agricultural Extension RAG: Smart Retrieval for Farmers"** (host: TRI AI).

Raw competition files (`documents.csv`, `train_queries.csv`, `qrels_train.csv`,
`test_queries.csv`, `sample_submission.csv`) are not redistributed here due to
Kaggle's competition data terms.

## To obtain the data

1. Join the competition on Kaggle and accept its rules.
2. Attach the competition dataset to a Kaggle Notebook via the Input panel, **or**
   download it locally with the Kaggle API:

   ```bash
   kaggle competitions download -c agricultural-extension-rag-smart-retrieval-for-farmers
   unzip agricultural-extension-rag-smart-retrieval-for-farmers.zip -d data/
   ```

See `scripts/agri_rag_retrieval_v7.ipynb` for how the data is loaded and used.
