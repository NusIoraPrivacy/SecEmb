from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def pad_or_trunc_indices(
    indices: np.ndarray,
    m_bar: int,
    num_items: int,
    rng,
) -> np.ndarray:
    """
    Pad or truncate a 1D array of item indices to length m_bar.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) > m_bar:
        # Randomly subsample
        idx = rng.choice(len(indices), size=m_bar, replace=False)
        return indices[idx]
    if len(indices) < m_bar:
        pad_size = m_bar - len(indices)
        pad = rng.integers(0, num_items, size=pad_size, dtype=np.int64)
        return np.concatenate([indices, pad], axis=0)
    return indices


def pad_or_trunc_rows(
    rows: torch.Tensor,
    m_bar: int,
    rng,
) -> torch.Tensor:
    """
    Pad or truncate a matrix of shape [k, dim] to [m_bar, dim].
    """
    if rows.numel() == 0:
        # All dummy rows
        return torch.zeros((m_bar, 0), dtype=rows.dtype, device=rows.device)

    k, dim = rows.shape
    if k > m_bar:
        idx = torch.from_numpy(
            rng.choice(k, size=m_bar, replace=False)
        ).to(rows.device)
        return rows[idx]
    if k < m_bar:
        pad_rows = torch.zeros(
            (m_bar - k, dim), dtype=rows.dtype, device=rows.device
        )
        return torch.cat([rows, pad_rows], dim=0)
    return rows

