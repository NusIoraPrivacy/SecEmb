from __future__ import annotations

"""
Unified SecEmb protocol module.

This file combines:
  - the original single-server, mask-based prototype (kept as a legacy
    example), and
  - the optimized 2-server FSS-style protocol from `optimized_protocol.py`,
    which is used for the main FL training simulation.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from .fss import CachedPathState, PointFSS, FSSKey, PartialKey
from .model import MatrixFactorizationModel, ModelConfig
from .padding import pad_or_trunc_indices, pad_or_trunc_rows


# ---------------------------------------------------------------------------
# Legacy single-server toy protocol (kept for reference)
# ---------------------------------------------------------------------------


@dataclass
class LegacyTrainingConfig:
    """
    Legacy configuration for the single-server toy protocol.
    """

    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.05
    sparsity_k: int = 50
    server_lr: float = 1.0


class SecureAggregator:
    """
    Toy secure aggregation for sparse embedding updates (legacy).
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    def client_mask_update(
        self,
        item_ids: torch.LongTensor,
        delta_embeddings: torch.Tensor,
        rng: np.random.Generator,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        assert delta_embeddings.ndim == 2
        num_items = delta_embeddings.shape[0]

        mask = torch.from_numpy(
            rng.normal(loc=0.0, scale=1.0, size=(num_items, self.embedding_dim))
        ).to(delta_embeddings.dtype)
        masked_delta = delta_embeddings + mask
        return item_ids, masked_delta, mask

    def aggregate_masked_updates(
        self,
        updates: Iterable[Tuple[torch.LongTensor, torch.Tensor]],
    ) -> Dict[int, torch.Tensor]:
        aggregated: Dict[int, torch.Tensor] = {}
        for item_ids, masked_delta in updates:
            for local_idx, item_id in enumerate(item_ids.tolist()):
                delta_vec = masked_delta[local_idx]
                if item_id not in aggregated:
                    aggregated[item_id] = delta_vec.clone()
                else:
                    aggregated[item_id] = aggregated[item_id] + delta_vec
        return aggregated

    def remove_aggregate_mask(
        self,
        aggregated: Dict[int, torch.Tensor],
        masks: Iterable[Tuple[torch.LongTensor, torch.Tensor]],
    ) -> Dict[int, torch.Tensor]:
        unmasked = {k: v.clone() for k, v in aggregated.items()}

        for item_ids, mask in masks:
            for local_idx, item_id in enumerate(item_ids.tolist()):
                mask_vec = mask[local_idx]
                if item_id not in unmasked:
                    unmasked[item_id] = -mask_vec.clone()
                else:
                    unmasked[item_id] = unmasked[item_id] - mask_vec

        return unmasked


class SecEmbServer:
    """
    Legacy single-server coordinator (kept for backwards compatibility).
    """

    def __init__(self, model_config: ModelConfig, train_config: LegacyTrainingConfig):
        self.device = torch.device("cpu")
        self.model = MatrixFactorizationModel(model_config).to(self.device)
        self.train_config = train_config
        self.sec_agg = SecureAggregator(embedding_dim=model_config.embedding_dim)

    @torch.no_grad()
    def get_item_embeddings(self, item_ids: torch.LongTensor) -> torch.Tensor:
        item_ids = item_ids.to(self.device)
        return self.model.get_item_embeddings(item_ids).cpu()

    @torch.no_grad()
    def apply_aggregated_sparse_update(
        self,
        sparse_update: Dict[int, torch.Tensor],
    ) -> None:
        if not sparse_update:
            return

        item_ids = torch.tensor(sorted(sparse_update.keys()), dtype=torch.long)
        delta_embeddings = torch.stack(
            [sparse_update[i] for i in item_ids.tolist()], dim=0
        )
        self.model.apply_sparse_item_update(
            item_ids=item_ids.to(self.device),
            delta_embeddings=delta_embeddings.to(self.device),
            lr=self.train_config.server_lr,
        )


class SecEmbClient:
    """
    Legacy single-server client (kept for backwards compatibility).
    """

    def __init__(
        self,
        client_id: int,
        user_id: int,
        num_items: int,
        interactions: Tuple[np.ndarray, np.ndarray],
        model_config: ModelConfig,
        train_config: LegacyTrainingConfig,
        rng: np.random.Generator,
    ):
        self.client_id = client_id
        self.user_id = user_id
        self.num_items = num_items
        self.model_config = model_config
        self.train_config = train_config
        self.rng = rng

        item_ids, ratings = interactions
        assert item_ids.shape == ratings.shape
        self.item_ids = torch.from_numpy(item_ids.astype(np.int64))
        self.ratings = torch.from_numpy(ratings.astype(np.float32))

        local_config = ModelConfig(
            num_users=model_config.num_users,
            num_items=model_config.num_items,
            embedding_dim=model_config.embedding_dim,
            user_reg=model_config.user_reg,
            item_reg=model_config.item_reg,
        )
        self.model = MatrixFactorizationModel(local_config)
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def sync_from_server(self, server: SecEmbServer) -> None:
        unique_items = torch.unique(self.item_ids)
        server_item_embeddings = server.get_item_embeddings(unique_items)
        with torch.no_grad():
            self.model.item_embeddings.weight[unique_items] = server_item_embeddings

    def local_train(self) -> Dict[int, torch.Tensor]:
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.train_config.lr
        )

        user_ids = torch.full_like(
            self.item_ids, fill_value=self.user_id, dtype=torch.long
        )
        dataset_size = self.item_ids.shape[0]

        for epoch in range(self.train_config.local_epochs):
            idx = self.rng.permutation(dataset_size)
            for start in range(0, dataset_size, self.train_config.batch_size):
                end = min(start + self.train_config.batch_size, dataset_size)
                batch_idx = idx[start:end]

                batch_users = user_ids[batch_idx].to(self.device)
                batch_items = self.item_ids[batch_idx].to(self.device)
                batch_ratings = self.ratings[batch_idx].to(self.device)

                optimizer.zero_grad()
                loss = self.model.loss(
                    user_ids=batch_users,
                    item_ids=batch_items,
                    ratings=batch_ratings,
                )
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            unique_items = torch.unique(self.item_ids)
            current_item_embeds = (
                self.model.item_embeddings.weight[unique_items].detach().clone()
            )

        delta = current_item_embeds
        norms = torch.norm(delta, dim=1)
        k = min(self.train_config.sparsity_k, delta.shape[0])
        if k <= 0:
            return {}

        _, topk_idx = torch.topk(norms, k=k, largest=True)
        selected_items = unique_items[topk_idx]
        selected_delta = delta[topk_idx]

        sparse_update = {
            int(item_id): selected_delta[i].detach().clone()
            for i, item_id in enumerate(selected_items.tolist())
        }
        return sparse_update

    def prepare_masked_sparse_update(
        self,
        sparse_update: Dict[int, torch.Tensor],
        aggregator: SecureAggregator,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        if not sparse_update:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, aggregator.embedding_dim),
                torch.empty(0, aggregator.embedding_dim),
            )

        item_ids = torch.tensor(sorted(sparse_update.keys()), dtype=torch.long)
        delta_embeddings = torch.stack(
            [sparse_update[i] for i in item_ids.tolist()], dim=0
        )
        return aggregator.client_mask_update(
            item_ids=item_ids,
            delta_embeddings=delta_embeddings,
            rng=self.rng,
        )


# ---------------------------------------------------------------------------
# Optimized 2-server FSS-style protocol (main API)
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.05
    m_bar: int = 50  # standardized per-user sparse support size
    server_lr: float = 1.0


@dataclass
class DenseState:
    """
    Aggregated dense gradient state on each server.
    """

    grad_sum: torch.Tensor


class SecEmbServerParty:
    """
    One of the two non-colluding servers for the optimized protocol.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainingConfig):
        self.device = torch.device("cpu")
        self.model = MatrixFactorizationModel(model_config).to(self.device)
        self.train_config = train_config
        self.fss = PointFSS(
            num_items=model_config.num_items,
            embedding_dim=model_config.embedding_dim,
        )

        dense_param = torch.zeros(1, dtype=torch.float32)
        self.dense_state = DenseState(grad_sum=dense_param.clone())

    # --- Retrieval phase ---

    def eval_retrieval(
        self,
        keys: List[FSSKey],
    ) -> List[torch.Tensor]:
        embs = self.model.item_embeddings.weight  # [num_items, dim]
        shares: List[torch.Tensor] = []
        for key in keys:
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
        assert len(partial_keys) == len(cached_paths)
        aggregated: Dict[int, torch.Tensor] = {}

        for pk, cache in zip(partial_keys, cached_paths):
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
    Run one full SecEmb round for a single user using the optimized protocol.
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

    k_real = min(len(rated_items), train_config.m_bar)
    local_item_embs = local_item_embs[:k_real]
    local_item_ids = target_indices[:k_real]

    # ---------- local training ----------
    local_config = ModelConfig(
        num_users=model_config.num_users,
        num_items=model_config.num_items,
        embedding_dim=model_config.embedding_dim,
        user_reg=model_config.user_reg,
        item_reg=model_config.item_reg,
    )
    local_model = MatrixFactorizationModel(local_config)

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
        for srv in (server0, server1):
            srv.model.apply_sparse_item_update(
                item_ids=item_ids_t,
                delta_embeddings=deltas,
                lr=train_config.server_lr,
            )

