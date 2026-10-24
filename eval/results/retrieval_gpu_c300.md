Questions: 80 (38 ar, 42 en) over 9 videos, 135 chunks of <= 300 chars. Embedding: `BAAI/bge-m3`, reranker: `BAAI/bge-reranker-v2-m3`.

| Retrieval | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Arabic Hit@5 | English Hit@5 | Latency (ms) |
|---|---|---|---|---|---|---|---|
| Dense (bge-m3) | 0.76 | 0.90 | 0.93 | 0.83 | 0.89 | 0.95 | 20 |
| Sparse (BM25) | 0.56 | 0.76 | 0.79 | 0.67 | 0.66 | 0.90 | 1 |
| Hybrid (RRF) | 0.71 | 0.90 | 0.96 | 0.81 | 0.95 | 0.98 | 17 |
| Hybrid + rerank | 0.86 | 0.97 | 1.00 | 0.92 | 1.00 | 1.00 | 97 |
