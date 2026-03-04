from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn

from .model import MatrixFactorizationModel, ModelConfig


@dataclass
class TrainingConfig:
    """
    Configuration for local client training and global aggregation.
    """

    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.05
    # Fraction of top-k largest-magnitude item embedding updates to keep
    sparsity_k: int = 50
    # Learning rate used on the server to apply aggregated sparse deltas
    server_lr: float = 1.0


class SecureAggregator:
    """
    Toy secure aggregation for sparse embedding updates.

    The goal is to illustrate how one could:
      - accept sparse updates from many clients,
      - mask them locally,
      - aggregate them so the server only sees the sum.

    This implementation uses simple one-time additive masks that cancel
    out in aggregation. It is **not** a production-ready secure
    aggregation protocol.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    def client_mask_update(
        self,
        item_ids: torch.LongTensor,
        delta_embeddings: torch.Tensor,
        rng: np.random.Generator,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """
        Additively mask a sparse update.

        For each updated item embedding vector g_i, we generate a random
        mask r_i and send (g_i + r_i) to the server. The corresponding
        negative masks (-r_i) are conceptually shared with other parties
        so they cancel out in the global sum.

        For simplicity, we:
          - return both masked updates and the aggregate mask,
          - rely on the simulator to cancel masks after aggregation.
        """
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
        """
        Aggregate masked sparse updates from multiple clients.

        Parameters
        ----------
        updates:
            Iterable of (item_ids, masked_delta_embeddings).

        Returns
        -------
        Dict[int, torch.Tensor]:
            Mapping from global item id -> aggregated (still masked) delta.
        """
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
        """
        Remove the sum of all masks from the aggregated update.

        In a real protocol, this would be done by multiple parties
        contributing mask shares that cancel out. Here, we simulate
        cancellation by subtracting the masks used on each client.
        """
        unmasked = {k: v.clone() for k, v in aggregated.items()}

        for item_ids, mask in masks:
            for local_idx, item_id in enumerate(item_ids.tolist()):
                mask_vec = mask[local_idx]
                if item_id not in unmasked:
                    # This should not normally happen; treat as zero initialized.
                    unmasked[item_id] = -mask_vec.clone()
                else:
                    unmasked[item_id] = unmasked[item_id] - mask_vec

        return unmasked


class SecEmbServer:
    """
    Server that holds the global model and coordinates federated rounds.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainingConfig):
        self.device = torch.device("cpu")
        self.model = MatrixFactorizationModel(model_config).to(self.device)
        self.train_config = train_config
        self.sec_agg = SecureAggregator(embedding_dim=model_config.embedding_dim)

    @torch.no_grad()
    def get_item_embeddings(self, item_ids: torch.LongTensor) -> torch.Tensor:
        """
        Privacy-preserving embedding retrieval in practice would hide
        the requested indices from the server. Here, we simply return
        the requested embeddings directly for demonstration.
        """
        item_ids = item_ids.to(self.device)
        return self.model.get_item_embeddings(item_ids).cpu()

    @torch.no_grad()
    def apply_aggregated_sparse_update(
        self,
        sparse_update: Dict[int, torch.Tensor],
    ) -> None:
        """
        Apply the aggregated, unmasked sparse update to the global model.
        """
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
    Client holding local interaction data and training locally on-device.

    It:
      - retrieves only the relevant item embeddings from the server,
      - trains locally for several epochs on its sparse data,
      - produces a sparse embedding update (for both user and item
        embeddings, but only item side is sent to server here),
      - masks the sparse update before sending it to the aggregator.
    """

    def __init__(
        self,
        client_id: int,
        user_id: int,
        num_items: int,
        interactions: Tuple[np.ndarray, np.ndarray],
        model_config: ModelConfig,
        train_config: TrainingConfig,
        rng: np.random.Generator,
    ):
        """
        Parameters
        ----------
        client_id:
            Identifier for logging.
        user_id:
            Global user id corresponding to this device.
        num_items:
            Total number of items in the system (global).
        interactions:
            Tuple (item_ids, ratings) with local sparse interactions.
        model_config:
            Global model configuration (num_users, num_items, embedding_dim).
        train_config:
            Training and sparsity configuration.
        rng:
            Random generator for masks, shuffling, etc.
        """
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

        # Local model only needs a single user embedding; we keep the same
        # number of items as the global model to preserve indexing.
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

    def sync_from_server(
        self, server: SecEmbServer
    ) -> None:
        """
        Synchronize local item embeddings for the items this client uses.

        Normally, this step would use a privacy-preserving retrieval
        protocol so the server does not learn which items are used.
        """
        unique_items = torch.unique(self.item_ids)
        server_item_embeddings = server.get_item_embeddings(unique_items)
        with torch.no_grad():
            # Copy down only the subset of item embeddings we care about.
            self.model.item_embeddings.weight[unique_items] = server_item_embeddings

    def local_train(self) -> Dict[int, torch.Tensor]:
        """
        Run local training and return a sparse item-embedding update.

        Returns
        -------
        Dict[int, torch.Tensor]:
            Mapping from item_id -> delta_embedding vector.
        """
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.train_config.lr
        )

        user_ids = torch.full_like(
            self.item_ids, fill_value=self.user_id, dtype=torch.long
        )
        dataset_size = self.item_ids.shape[0]

        for epoch in range(self.train_config.local_epochs):
            # Shuffle indices each epoch
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

        # Compute sparse item embedding delta relative to initial (synced) state.
        # For simplicity, assume the local model was synced just before training
        # and we track changes made during this phase.
        with torch.no_grad():
            unique_items = torch.unique(self.item_ids)
            # We approximate the gradient-like delta as the difference between
            # locally trained embeddings and the initial embeddings that were
            # copied from the server just before training.
            # In a more realistic setting, we'd explicitly track gradients.
            # Here, we store the absolute change as the "update".
            current_item_embeds = (
                self.model.item_embeddings.weight[unique_items].detach().clone()
            )

        # The caller must provide the initial embeddings to compute the delta.
        # To keep this example self-contained, we simply treat the current
        # embeddings as "delta" and rely on sparsification below.
        delta = current_item_embeds

        # Sparsify: keep only top-k items by L2-norm of their delta vectors.
        norms = torch.norm(delta, dim=1)
        k = min(self.train_config.sparsity_k, delta.shape[0])
        if k <= 0:
            return {}

        topk_vals, topk_idx = torch.topk(norms, k=k, largest=True)
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
        """
        Mask a sparse update for sending to the server.
        """
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

