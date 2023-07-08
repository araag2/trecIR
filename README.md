# TREC-IR
Repository created to train methods on the TREC-IR dataset, specifically in tracks related to medical information.

# Generate pyserini indexes

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input TREC2021/base_indexes/free_txt/ \
  --index TREC2021/indexes/free_txt/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storeDocvectors 