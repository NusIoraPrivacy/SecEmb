from __future__ import annotations

"""
Optimized SecEmb protocol (2-server, FSS-style, row-wise upload with
path reuse) following the design in `Alg.md`.

This module uses a mock point-function FSS backend (`fss.PointFSS`)
which is NOT cryptographically secure but exposes the same APIs
(`gen_with_cache`, `eval`, `convert_gen`, `convert_eval`) needed to
structure the protocol correctly.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from .fss import CachedPathState, PointFSS, FSSKey, PartialKey
from .model import MatrixFactorizationModel, ModelConfig
from .padding import pad_or_trunc_indices, pad_or_trunc_rows


@dataclass
class DenseState:
    """
    Aggregated dense gradient state on each server.
    """

    grad_sum: torch.Tensor


@dataclass
class TrainingConfig:
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.05
    m_bar: int = 50  # standardized per-user sparse support size
    server_lr: float = 1.0


class SecEmbServerParty:
    """
    One of the two non-colluding servers.

    Each party holds:
      - a share of the global item embeddings,
      - a share of the dense parameters,
      - FSS evaluation state for retrieval and sparse upload.

    For simplicity in this prototype:
      - each party holds the full cleartext model,
      - FSS is only used to structure indices/payloads,
      - dense gradients are additive shares.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainingConfig):
        self.device = torch.device("cpu")
        self.model = MatrixFactorizationModel(model_config).to(self.device)
        self.train_config = train_config
        self.fss = PointFSS()

        # Dense parameter accumulator (single scalar for MF bias-free model).
        dense_param = torch.zeros(1, dtype=torch.float32)
        self.dense_state = DenseState(grad_sum=dense_param.clone())

    # --- Retrieval phase ---

    def eval_retrieval(
        self,
        keys: List[FSSKey],
    ) -> List[torch.Tensor]:
        """
        Evaluate retrieval keys over the item embedding table.

        For each key, compute:
            share_emb = sum_x Eval(key, x) * ItemEmb[x]

        In the mock backend, Eval(key, x) is non-zero only at x = alpha.
        """
        embs = self.model.item_embeddings.weight  # [num_items, dim]
        shares: List[torch.Tensor] = []
        for key in keys:
            # Only need to evaluate at x = key.alpha in this fake backend.
            coeff = self.fss.eval(key, key.alpha)  # payload is scalar 1
            share = coeff * embs[key.alpha]
            shares.append(share.detach().clone())
        return shares

    # --- Sparse upload phase ---

    def aggregate_sparse(
        self,
        partial_keys: List[PartialKey],
        cached_paths: List[CachedPathState],
    ) -> Dict[int, torch.Tensor]:
        """
        Aggregate row-wise sparse updates using cached paths.
        """
        assert len(partial_keys) == len(cached_paths)
        aggregated: Dict[int, torch.Tensor] = {}

        for pk, cache in zip(partial_keys, cached_paths):
            # Evaluate at x = alpha only; in real FSS this would reuse
            # cached path-evaluation state along the tree.
            row_share = self.fss.convert_eval(cache, pk, cache.alpha)
            if cache.alpha not in aggregated:
                aggregated[cache.alpha] = row_share.clone()
            else:
                aggregated[cache.alpha] = aggregated[cache.alpha] + row_share

        return aggregated

    # --- Dense upload phase ---

    def add_dense_share(self, share: torch.Tensor) -> None:
        self.dense_state.grad_sum = self.dense_state.grad_sum + share


# --- Helper functions for the user / client side ---


def user_prepare_retrieval_keys(
    rated_items: np.ndarray,
    num_items: int,
    m_bar: int,
    fss: PointFSS,
    rng,
) -> Tuple[List[FSSKey], List[FSSKey], List[CachedPathState], np.ndarray]:
    """
    Standardize and encode indices for the retrieval phase.
    """
    target_indices = pad_or_trunc_indices(rated_items, m_bar, num_items, rng)
    keys0: List[FSSKey] = []
    keys1: List[FSSKey] = []
    cached_paths: List[CachedPathState] = []

    for item_id in target_indices:
        alpha = int(item_id)
        beta = torch.tensor(1.0)  # scalar payload 1
        k0, k1, cache = fss.gen_with_cache(alpha=alpha, beta=beta, rng=rng)
        keys0.append(k0)
        keys1.append(k1)
        cached_paths.append(cache)

    return keys0, keys1, cached_paths, target_indices


def user_reconstruct_embeddings(
    shares0: List[torch.Tensor],
    shares1: List[torch.Tensor],
) -> torch.Tensor:
    """
    Reconstruct full item embeddings for the user's target indices.
    """
    assert len(shares0) == len(shares1)
    embs = []
    for s0, s1 in zip(shares0, shares1):
        embs.append(s0 + s1)
    return torch.stack(embs, dim=0)


def share_dense_gradient(
    dense_grad: torch.Tensor,
    rng,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-server additive sharing of dense gradient.
    """
    share0 = torch.randn_like(
        dense_grad, generator=torch.Generator().manual_seed(int(rng.integers(1, 1_000_000)))
    )
    share1 = dense_grad - share0
    return share0, share1


def secemb_round(
    user_id: int,
    rated_items: np.ndarray,
    ratings: np.ndarray,
    server0: SecEmbServerParty,
    server1: SecEmbServerParty,
    model_config: ModelConfig,
    train_config: TrainingConfig,
    rng,
) -> None:
    """
    Run one full SecEmb round for a single user.

    This includes:
      - retrieval (FSS-based),
      - local training,
      - row-wise sparse upload with path reuse,
      - dense additive-share upload,
      - server-side aggregation and global model update.
    """
    num_items = model_config.num_items
    fss = server0.fss  # both parties use same backend instance type

    # ---------- retrieval ----------
    keys0, keys1, cached_paths, target_indices = user_prepare_retrieval_keys(
        rated_items, num_items, train_config.m_bar, fss, rng
    )
    shares0 = server0.eval_retrieval(keys0)
    shares1 = server1.eval_retrieval(keys1)
    local_item_embs = user_reconstruct_embeddings(shares0, shares1)

    # Discard dummy embeddings locally by keeping only positions
    # corresponding to real rated_items (first len(rated_items)).
    k_real = min(len(rated_items), train_config.m_bar)
    local_item_embs = local_item_embs[:k_real]
    local_item_ids = target_indices[:k_real]

    # ---------- local training ----------
    # Build a small local MF model that uses the retrieved item embeddings.
    local_config = ModelConfig(
        num_users=model_config.num_users,
        num_items=model_config.num_items,
        embedding_dim=model_config.embedding_dim,
        user_reg=model_config.user_reg,
        item_reg=model_config.item_reg,
    )
    local_model = MatrixFactorizationModel(local_config)

    # Initialize only the relevant item rows
    with torch.no_grad():
        local_model.item_embeddings.weight[torch.from_numpy(local_item_ids)] = (
            local_item_embs
        )

    optimizer = torch.optim.SGD(local_model.parameters(), lr=train_config.lr)
    local_model.train()

    user_ids = torch.full(
        (len(rated_items),), fill_value=user_id, dtype=torch.long
    )
    item_ids = torch.from_numpy(rated_items.astype(np.int64))
    ratings_t = torch.from_numpy(ratings.astype(np.float32))

    dataset_size = len(rated_items)
    for epoch in range(train_config.local_epochs):
        idx = rng.permutation(dataset_size)
        for start in range(0, dataset_size, train_config.batch_size):
            end = min(start + train_config.batch_size, dataset_size)
            batch_idx = idx[start:end]
            batch_users = user_ids[batch_idx]
            batch_items = item_ids[batch_idx]
            batch_ratings = ratings_t[batch_idx]

            optimizer.zero_grad()
            loss = local_model.loss(
                user_ids=batch_users,
                item_ids=batch_items,
                ratings=batch_ratings,
            )
            loss.backward()
            optimizer.step()

    # Compute sparse item embedding gradient approximation as row differences
    with torch.no_grad():
        updated_item_embs = local_model.item_embeddings.weight[
            torch.from_numpy(local_item_ids)
        ]
        sparse_rows = updated_item_embs - local_item_embs

    # Standardize number of sparse rows
    sparse_rows = pad_or_trunc_rows(
        sparse_rows, train_config.m_bar, rng=rng
    )  # [m_bar, dim]

    # ---------- sparse upload via path reuse ----------
    partial_keys0: List[PartialKey] = []
    partial_keys1: List[PartialKey] = []

    for cache, row_grad in zip(cached_paths, sparse_rows):
        pk0, pk1 = fss.convert_gen(cache, beta=row_grad, rng=rng)
        partial_keys0.append(pk0)
        partial_keys1.append(pk1)

    sparse_share0 = server0.aggregate_sparse(partial_keys0, cached_paths)
    sparse_share1 = server1.aggregate_sparse(partial_keys1, cached_paths)

    # ---------- dense upload ----------
    dense_grad = torch.zeros(1, dtype=torch.float32)  # MF has no extra dense params
    d0, d1 = share_dense_gradient(dense_grad, rng)
    server0.add_dense_share(d0)
    server1.add_dense_share(d1)

    # ---------- server combine and apply update ----------
    # Combine sparse shares
    aggregated_sparse: Dict[int, torch.Tensor] = {}
    for k, v in sparse_share0.items():
        aggregated_sparse[k] = v.clone()
    for k, v in sparse_share1.items():
        if k in aggregated_sparse:
            aggregated_sparse[k] = aggregated_sparse[k] + v
        else:
            aggregated_sparse[k] = v.clone()

    if aggregated_sparse:
        item_ids_t = torch.tensor(sorted(aggregated_sparse.keys()), dtype=torch.long)
        deltas = torch.stack(
            [aggregated_sparse[int(i)] for i in item_ids_t.tolist()], dim=0
        )
        # Apply update to both parties' models (they hold cleartext here)
        for srv in (server0, server1):
            srv.model.apply_sparse_item_update(
                item_ids=item_ids_t,
                delta_embeddings=deltas,
                lr=train_config.server_lr,
            )

