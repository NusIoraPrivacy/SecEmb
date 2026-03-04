from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class ModelConfig:
    num_users: int
    num_items: int
    embedding_dim: int = 32
    user_reg: float = 1e-4
    item_reg: float = 1e-4


class MatrixFactorizationModel(nn.Module):
    """
    Simple embedding-based recommender (matrix factorization).

    - Users and items are represented by embedding vectors.
    - Predictions are dot products of user and item embeddings.
    - Loss is MSE on observed interactions plus L2 regularization.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.user_embeddings = nn.Embedding(
            num_embeddings=config.num_users,
            embedding_dim=config.embedding_dim,
        )
        self.item_embeddings = nn.Embedding(
            num_embeddings=config.num_items,
            embedding_dim=config.embedding_dim,
        )

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(
        self,
        user_ids: torch.LongTensor,
        item_ids: torch.LongTensor,
    ) -> torch.Tensor:
        u = self.user_embeddings(user_ids)
        v = self.item_embeddings(item_ids)
        # Dot product along embedding dimension
        return (u * v).sum(dim=-1)

    def loss(
        self,
        user_ids: torch.LongTensor,
        item_ids: torch.LongTensor,
        ratings: torch.Tensor,
    ) -> torch.Tensor:
        preds = self.forward(user_ids, item_ids)
        mse = torch.mean((preds - ratings) ** 2)

        reg = 0.0
        if self.config.user_reg > 0.0:
            reg = reg + self.config.user_reg * torch.sum(
                self.user_embeddings.weight ** 2
            )
        if self.config.item_reg > 0.0:
            reg = reg + self.config.item_reg * torch.sum(
                self.item_embeddings.weight ** 2
            )
        return mse + reg

    @torch.no_grad()
    def get_item_embeddings(self, item_ids: torch.LongTensor) -> torch.Tensor:
        return self.item_embeddings(item_ids).detach().clone()

    @torch.no_grad()
    def set_item_embeddings(
        self,
        item_ids: torch.LongTensor,
        new_embeddings: torch.Tensor,
    ) -> None:
        """
        In-place update of a subset of item embeddings.
        """
        self.item_embeddings.weight[item_ids] = new_embeddings

    @torch.no_grad()
    def apply_sparse_item_update(
        self,
        item_ids: torch.LongTensor,
        delta_embeddings: torch.Tensor,
        lr: float,
    ) -> None:
        """
        Apply sparse gradient-style updates to a subset of item embeddings.
        """
        self.item_embeddings.weight[item_ids] += lr * delta_embeddings

