Questions: 80 (38 ar, 42 en) over 9 videos, 43 chunks of <= 800 chars. Embedding: `BAAI/bge-m3`, reranker: `BAAI/bge-reranker-v2-m3`.

| Retrieval | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Arabic Hit@5 | English Hit@5 | Latency (ms) |
|---|---|---|---|---|---|---|---|
| Dense (bge-m3) | 0.80 | 0.94 | 0.99 | 0.87 | 0.97 | 1.00 | 20 |
| Sparse (BM25) | 0.66 | 0.85 | 0.89 | 0.76 | 0.76 | 1.00 | 1 |
| Hybrid (RRF) | 0.79 | 0.96 | 0.99 | 0.87 | 0.97 | 1.00 | 18 |
| Hybrid + rerank | 0.93 | 0.99 | 1.00 | 0.96 | 1.00 | 1.00 | 197 |
