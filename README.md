# Project: Multimodal AI Video Q&A System

**Tech Stack:** FastAPI, MongoDB, Qdrant, SentenceTransformers, LLaMA 3 1B (local)

## Description / Key Contributions

- Built an end-to-end pipeline to ingest YouTube videos, extract audio, transcribe speech, and chunk text for semantic search.  
- Implemented vector embeddings storage in **Qdrant** and retrieval-augmented generation (RAG) using **LLaMA 3** for local query answering.  
- Designed **FastAPI** endpoints for asynchronous processing of user queries and real-time responses.  
- Leveraged open-source LLMs and embeddings for cost-effective, scalable AI-powered search over video content.
