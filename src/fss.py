from __future__ import annotations

"""
Mock point-function FSS with path caching, tailored to SecEmb.

This is a **prototype** implementation that follows the API suggested
in `Alg.md` but does NOT provide real cryptographic security. It is
designed so that the rest of the SecEmb protocol (retrieval + row-wise
upload with path reuse) can be implemented and tested end-to-end.
"""

from dataclasses import dataclass
from typing import Any, Tuple

import torch


@dataclass
class CachedPathState:
    """
    Cached state for a point-function index path.

    In a real FSS backend this would store seeds / correction words
    along the binary tree path. Here we just keep the index `alpha`.
    """

    alpha: int


@dataclass
class FSSKey:
    """
    Full FSS key for a point function (mock).

    For the prototype we only store:
      - alpha: point index
      - beta_share: this party's payload share (scalar or vector)
    """

    alpha: int
    beta_share: torch.Tensor


@dataclass
class PartialKey:
    """
    Partial key for the optimized upload phase.

    This is the "conversion" part that carries only the payload share.
    The index-dependent path is reused from `CachedPathState`.
    """

    beta_share: torch.Tensor


class PointFSS:
    """
    Simple 2-party additive point-function "FSS" over tensors.

    Semantics:
      - Domain: integer item indices x in [0, num_items)
      - Payload: scalar or vector encoded as a torch.Tensor
      - Two parties hold shares such that:

          Eval(k0, x) + Eval(k1, x) = f(x)

        where f(x) = beta if x == alpha else 0.

    NOTE: This is NOT a real FSS protocol; it only emulates the API.
    """

    def gen_with_cache(
        self,
        alpha: int,
        beta: torch.Tensor,
        rng,
    ) -> Tuple[FSSKey, FSSKey, CachedPathState]:
        """
        Generate two additive shares of a point function with caching.
        """
        # Sample random share for party 0 and derive party 1's share
        beta0 = torch.randn_like(beta, generator=torch.Generator().manual_seed(int(rng.integers(1, 1_000_000)))).to(
            beta.dtype
        )
        beta1 = beta - beta0
        k0 = FSSKey(alpha=alpha, beta_share=beta0)
        k1 = FSSKey(alpha=alpha, beta_share=beta1)
        cache = CachedPathState(alpha=alpha)
        return k0, k1, cache

    def eval(self, key: FSSKey, x: int) -> torch.Tensor:
        """
        Evaluate this party's share at point x.
        """
        if x == key.alpha:
            return key.beta_share
        # Return a zero tensor with same shape/dtype as payload share
        return torch.zeros_like(key.beta_share)

    # --- Optimized upload path: reuse CachedPathState ---

    def convert_gen(
        self,
        cache: CachedPathState,
        beta: torch.Tensor,
        rng,
    ) -> Tuple[PartialKey, PartialKey]:
        """
        Derive partial keys for a new payload using the cached path.

        In a real FSS, this would only generate payload-specific tail
        words. Here we again sample an additive share decomposition.
        """
        beta0 = torch.randn_like(beta, generator=torch.Generator().manual_seed(int(rng.integers(1, 1_000_000)))).to(
            beta.dtype
        )
        beta1 = beta - beta0
        return PartialKey(beta_share=beta0), PartialKey(beta_share=beta1)

    def convert_eval(
        self,
        cache: CachedPathState,
        partial_key: PartialKey,
        x: int,
    ) -> torch.Tensor:
        """
        Evaluate this party's share for the upload phase, reusing path state.
        """
        if x == cache.alpha:
            return partial_key.beta_share
        return torch.zeros_like(partial_key.beta_share)

