"""
SecEmb prototype package.

This package provides a minimal, educational implementation inspired by:

  SecEmb: Sparsity-Aware Secure Federated Learning of On-Device
  Recommender System with Large Embedding
  (arXiv:2505.12453, https://arxiv.org/abs/2505.12453)

It contains:
  - A simple embedding-based recommender model.
  - A sparsity-aware federated client that trains locally.
  - A toy secure aggregation protocol for sparse embedding updates.

This code is **not** a production-ready or cryptographically sound
implementation. It is meant to illustrate the high-level structure
and data flow described in the paper.
"""

from .model import MatrixFactorizationModel
from .protocol import (
    SecEmbServer,
    SecEmbClient,
    SecureAggregator,
    TrainingConfig,
)

__all__ = [
    "MatrixFactorizationModel",
    "SecEmbServer",
    "SecEmbClient",
    "SecureAggregator",
    "TrainingConfig",
]

