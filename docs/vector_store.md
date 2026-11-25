# vector_store.py

This module wraps ChromaDB and the embedding model into two clean operations: create and load. That separation matters because in the ingestion workflow you create a new store, but in a production query workflow you load an existing persisted one.

We use `HuggingFaceEmbeddings` from `langchain-huggingface` with the `all-MiniLM-L6-v2` model. This model runs locally — no API call, no cost per embedding, and no network latency on every document. It's a solid general-purpose embedding model: small enough to be fast, capable enough to handle the semantic search we need. The embeddings run once at index time, so the only ongoing cost is the retrieval query embedding, which is negligible.

The `collection_metadata={"hnsw:space": "cosine"}` setting tells ChromaDB to use cosine similarity rather than L2 distance for its HNSW index. Cosine similarity is more appropriate for semantic search because it measures directional alignment between embeddings rather than absolute magnitude — two chunks that are semantically similar but have different text lengths will still score highly with cosine, whereas L2 would penalize them for being different magnitudes.

The `_embedding_model` helper is extracted as a private function to avoid repeating the model instantiation string in both `create` and `load`. Both functions accept an optional `persist_directory` that falls back to the config default, which keeps the API flexible for tests or multi-tenant scenarios where you might want to point at a different store without touching the config.
