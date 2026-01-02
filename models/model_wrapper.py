"""
Model wrapper providing unified interface for embedding models.
Author: chandu
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch


class EmbeddingModel(ABC):
    """Abstract base class for all embedding models."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of strings to encode

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the model name."""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        pass


class LocalModel(EmbeddingModel):
    """Wrapper for local sentence-transformer models."""

    def __init__(self, model_name: str, model_id: str, dimensions: int):
        """
        Initialize local model.

        Args:
            model_name: HuggingFace model identifier
            model_id: Short ID for the model
            dimensions: Embedding dimension
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model_id = model_id
        self.dimensions = dimensions

        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {model_id} ({model_name}) on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of strings to encode
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings (for cosine similarity)

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        # Encode using sentence-transformers
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embeddings

    def get_name(self) -> str:
        """Get the model ID."""
        return self.model_id

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.dimensions

    def get_full_name(self) -> str:
        """Get the full model name."""
        return self.model_name
