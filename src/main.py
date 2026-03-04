"""
Small end-to-end simulation of the SecEmb-style federated protocol.

This script:
  - creates a global embedding-based recommender,
  - creates several synthetic clients with sparse item interactions,
  - runs multiple federated rounds where clients:
      * retrieve relevant item embeddings from the server,
      * train locally on-device,
      * produce sparse item embedding updates,
      * mask these updates and send them to the server via a toy
        secure aggregation mechanism,
  - applies the aggregated sparse update on the server model.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from .model import ModelConfig
from .protocol import SecEmbServerParty, TrainingConfig, secemb_round


def load_movielens_100k(
    root: str = "data/ml-100k",
    filename: str = "u.data",
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int, int]:
    """
    Load MovieLens 100K data from the `data` directory and group
    interactions by user.

    MovieLens 100K format (u.data):
        user_id\\titem_id\\trating\\ttimestamp

    Returns
    -------
    client_data:
        List indexed by (zero-based) user index; each element is
        (item_ids, ratings) as numpy arrays.
    num_users:
        Total number of distinct users.
    num_items:
        Total number of distinct items.
    """
    path = os.path.join(root, filename)
    user_item_ratings: Dict[int, List[Tuple[int, float]]] = {}
    max_user = 0
    max_item = 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            user_id_raw, item_id_raw, rating_raw = parts[:3]
            user_id = int(user_id_raw) - 1  # convert to 0-based
            item_id = int(item_id_raw) - 1  # convert to 0-based
            rating = float(rating_raw)

            if user_id not in user_item_ratings:
                user_item_ratings[user_id] = []
            user_item_ratings[user_id].append((item_id, rating))

            max_user = max(max_user, user_id)
            max_item = max(max_item, item_id)

    num_users = max_user + 1
    num_items = max_item + 1

    client_data: List[Tuple[np.ndarray, np.ndarray]] = []
    for u in range(num_users):
        interactions = user_item_ratings.get(u, [])
        if not interactions:
            client_data.append(
                (np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32))
            )
            continue
        items, ratings = zip(*interactions)
        client_data.append(
            (np.asarray(items, dtype=np.int64), np.asarray(ratings, dtype=np.float32))
        )

    return client_data, num_users, num_items


def evaluate_global_loss(
    model: torch.nn.Module,
    data: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    """
    Evaluate the global model loss on the union of all user interactions.
    """
    model.eval()

    losses = []
    with torch.no_grad():
        for user_id, (item_ids_np, ratings_np) in enumerate(data):
            if len(item_ids_np) == 0:
                continue
            user_ids = torch.full(
                (len(item_ids_np),),
                fill_value=user_id,
                dtype=torch.long,
            )
            item_ids = torch.from_numpy(item_ids_np.astype(np.int64))
            ratings = torch.from_numpy(ratings_np.astype(np.float32))
            loss = model.loss(user_ids, item_ids, ratings)
            losses.append(loss.item())

    return float(sum(losses) / max(len(losses), 1))


def run_simulation() -> None:
    rng = np.random.default_rng(seed=42)

    # Load real data from data/ml-100k/u.data
    client_data, num_users, num_items = load_movielens_100k()

    model_config = ModelConfig(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=32,
        user_reg=1e-4,
        item_reg=1e-4,
    )
    train_config = TrainingConfig(
        local_epochs=1,
        batch_size=128,
        lr=0.05,
        m_bar=50,
        server_lr=1.0,
    )

    # Initialize two non-colluding server parties
    server0 = SecEmbServerParty(model_config=model_config, train_config=train_config)
    server1 = SecEmbServerParty(model_config=model_config, train_config=train_config)

    # Evaluate initial global loss (servers start identical)
    initial_loss = evaluate_global_loss(server0.model, client_data)
    print(f"Initial global loss: {initial_loss:.4f}")

    num_rounds = 3
    for round_idx in range(num_rounds):
        print(f"\n=== SecEmb round {round_idx + 1}/{num_rounds} ===")

        for user_id, (items_np, ratings_np) in enumerate(client_data):
            if len(items_np) == 0:
                continue
            user_rng = np.random.default_rng(seed=rng.integers(0, 1_000_000))
            secemb_round(
                user_id=user_id,
                rated_items=items_np,
                ratings=ratings_np,
                server0=server0,
                server1=server1,
                model_config=model_config,
                train_config=train_config,
                rng=user_rng,
            )

        # Evaluate global loss after this round
        loss = evaluate_global_loss(server0.model, client_data)
        print(f"Global loss after round {round_idx + 1}: {loss:.4f}")


if __name__ == "__main__":
    run_simulation()

