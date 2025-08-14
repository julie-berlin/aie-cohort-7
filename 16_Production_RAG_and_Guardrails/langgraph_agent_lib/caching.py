"""Production caching utilities for embeddings and LLM calls."""

import hashlib
import os
from typing import Optional

from langchain.embeddings import CacheBackedEmbeddings as LangChainCacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_openai.embeddings import OpenAIEmbeddings


class CacheBackedEmbeddings:
    """Production cache-backed embeddings using OpenAI."""
    
    def __init__(
        self, 
        model: str = "text-embedding-3-small",
        cache_dir: str = "./cache/embeddings",
        batch_size: int = 32
    ):
        """Initialize cache-backed embeddings.
        
        Args:
            model: OpenAI embedding model name
            cache_dir: Directory to store embedding cache
            batch_size: Batch size for embedding calls
        """
        self.model = model
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        
        # Create base embeddings
        self.base_embeddings = OpenAIEmbeddings(model=model)
        
        # Create safe namespace from model name
        safe_namespace = hashlib.md5(model.encode()).hexdigest()
        
        # Set up file store and cached embeddings
        store = LocalFileStore(cache_dir)
        self.cached_embeddings = LangChainCacheBackedEmbeddings.from_bytes_store(
            self.base_embeddings, 
            store, 
            namespace=safe_namespace,
            batch_size=batch_size
        )
    
    def get_embeddings(self):
        """Get the cached embeddings instance."""
        return self.cached_embeddings


def setup_llm_cache(cache_type: str = "memory", cache_path: Optional[str] = None):
    """Set up LLM caching.
    
    Args:
        cache_type: Type of cache - "memory" or "sqlite"
        cache_path: Path for SQLite cache file
    """
    if cache_type == "memory":
        set_llm_cache(InMemoryCache())
    elif cache_type == "sqlite":
        db_path = cache_path or "./cache/llm_cache.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=db_path))
    else:
        raise ValueError(f"Unsupported cache type: {cache_type}")

