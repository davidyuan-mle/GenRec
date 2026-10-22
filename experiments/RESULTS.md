# Results Log

Amazon Beauty (5-core), leave-one-out split, full-catalog ranking unless noted.

| Date | Milestone | Method | Recall@10 | NDCG@10 | MRR | Notes |
|---|---|---|---|---|---|---|
| 2026-10-21 | 1 | Popularity | 0.0114 | 0.0053 | 0.0040 | test split; run popularity_beauty-20261021-231939 |
| 2026-10-22 | 1 | SASRec | 0.0753 | 0.0423 | 0.0346 | test split; 50 epochs, best epoch 50 (valid NDCG@10 plateaued from ~30); run sasrec_beauty-20261022-000854 |
| 2026-10-22 | 2 | Semantic kNN (sentence-t5-base) | 0.0593 | 0.0350 | 0.0290 | test split; no training, n_recent=5, decay=0.5; run semantic_knn_beauty-20261022-141411 |
