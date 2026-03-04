from __future__ import annotations

"""
SecEmb FSS components and Torch adapter.

This module combines:
  - a reference implementation of the SecEmb point-function FSS over
    G = Z_q^d (adapted from ``fss_new.py``), and
  - a Torch-facing backend that exposes the high-level API used by the
    optimized federated training protocol:

        gen_with_cache(alpha, beta, rng) -> (FSSKey, FSSKey, CachedPathState)
        eval(key, x) -> torch.Tensor
        convert_gen(cache, beta, rng) -> (PartialKey, PartialKey)
        convert_eval(cache, partial_key, x) -> torch.Tensor
"""

from dataclasses import dataclass
from hashlib import sha256
import math
import secrets
from typing import List, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Core SecEmb FSS over Z_q^d
# ---------------------------------------------------------------------------

Vec = List[int]


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise ValueError("byte strings must have equal length")
    return bytes(x ^ y for x, y in zip(a, b))


def _sha256(data: bytes) -> bytes:
    return sha256(data).digest()


@dataclass(frozen=True)
class CoreCorrectionWord:
    s_cw: bytes
    t_l_cw: int
    t_r_cw: int


@dataclass(frozen=True)
class CoreFSSKey:
    seed0: bytes
    t0: int
    cws: Tuple[CoreCorrectionWord, ...]
    final_cw: Tuple[int, ...]


@dataclass(frozen=True)
class CorePathState:
    t_n_1: int
    s_n_0: bytes
    s_n_1: bytes
    cws: Tuple[CoreCorrectionWord, ...]
    n_bits: int


class SecEmbFSS:
    """
    Reference SecEmb point-function FSS over G = Z_q^d.
    """

    def __init__(self, lambda_bits: int = 128, modulus: int = 2**61 - 1, dim: int = 1):
        if lambda_bits % 8 != 0:
            raise ValueError("lambda_bits must be divisible by 8")
        if modulus <= 2:
            raise ValueError("modulus must be > 2")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.lambda_bits = lambda_bits
        self.seed_bytes = lambda_bits // 8
        self.modulus = modulus
        self.dim = dim

    # ---------- Group helpers for G = Z_q^d ----------
    def zero(self) -> Vec:
        return [0] * self.dim

    def add(self, a: Sequence[int], b: Sequence[int]) -> Vec:
        return [int((x + y) % self.modulus) for x, y in zip(a, b)]

    def sub(self, a: Sequence[int], b: Sequence[int]) -> Vec:
        return [int((x - y) % self.modulus) for x, y in zip(a, b)]

    def neg(self, a: Sequence[int]) -> Vec:
        return [int((-x) % self.modulus) for x in a]

    def scalar_bit_mul(self, bit: int, a: Sequence[int]) -> Vec:
        return [int((bit * x) % self.modulus) for x in a]

    # ---------- PRG and Convert_G ----------
    def _expand_seed(self, seed: bytes, domain_sep: bytes) -> bytes:
        out = b""
        counter = 0
        while len(out) < self.seed_bytes:
            out += _sha256(domain_sep + seed + counter.to_bytes(4, "big"))
            counter += 1
        return out[: self.seed_bytes]

    def convert_g(self, seed: bytes) -> Vec:
        """Convert_G: {0,1}^lambda -> G = Z_q^d."""
        out: Vec = []
        counter = 0
        while len(out) < self.dim:
            block = _sha256(b"CONVERT" + seed + counter.to_bytes(4, "big"))
            out.append(int.from_bytes(block, "big") % self.modulus)
            counter += 1
        return out

    def prg(self, seed: bytes) -> Tuple[bytes, int, bytes, int]:
        """G: {0,1}^lambda -> {0,1}^{2(lambda+1)}.

        Returns (s_L, t_L, s_R, t_R).
        """
        s_l = self._expand_seed(seed, b"PRG-L-SEED")
        s_r = self._expand_seed(seed, b"PRG-R-SEED")
        t_l = _sha256(b"PRG-L-BIT" + seed)[0] & 1
        t_r = _sha256(b"PRG-R-BIT" + seed)[0] & 1
        return s_l, t_l, s_r, t_r

    # ---------- Bit utilities ----------
    @staticmethod
    def _int_to_bits(x: int, n_bits: int) -> List[int]:
        if x < 0 or x >= (1 << n_bits):
            raise ValueError("x out of range for n_bits")
        return [(x >> shift) & 1 for shift in range(n_bits - 1, -1, -1)]

    @staticmethod
    def bits_for_domain(domain_size: int) -> int:
        if domain_size <= 1:
            return 1
        return math.ceil(math.log2(domain_size))

    # ---------- Algorithms 5 + 7 combined: Gen + path cache ----------
    def gen(
        self,
        alpha: int,
        beta: Sequence[int],
        *,
        domain_size: int | None = None,
        n_bits: int | None = None,
    ) -> Tuple[CoreFSSKey, CoreFSSKey, CorePathState]:
        if n_bits is None:
            if domain_size is None:
                raise ValueError("provide either n_bits or domain_size")
            n_bits = self.bits_for_domain(domain_size)

        alpha_bits = self._int_to_bits(alpha, n_bits)

        s0_init = secrets.token_bytes(self.seed_bytes)
        s1_init = secrets.token_bytes(self.seed_bytes)
        t0_init = secrets.randbits(1)
        t1_init = t0_init ^ 1

        s0 = s0_init
        s1 = s1_init
        t0 = t0_init
        t1 = t1_init
        cws: List[CoreCorrectionWord] = []

        for alpha_i in alpha_bits:
            s_l0, t_l0, s_r0, t_r0 = self.prg(s0)
            s_l1, t_l1, s_r1, t_r1 = self.prg(s1)

            if alpha_i == 0:
                keep = "L"
                lose = "R"
            else:
                keep = "R"
                lose = "L"

            s_lose_0 = s_r0 if lose == "R" else s_l0
            s_lose_1 = s_r1 if lose == "R" else s_l1
            s_cw = _xor_bytes(s_lose_0, s_lose_1)

            t_l_cw = t_l0 ^ t_l1 ^ alpha_i ^ 1
            t_r_cw = t_r0 ^ t_r1 ^ alpha_i
            cws.append(
                CoreCorrectionWord(s_cw=s_cw, t_l_cw=t_l_cw, t_r_cw=t_r_cw)
            )

            s_keep_0 = s_l0 if keep == "L" else s_r0
            s_keep_1 = s_l1 if keep == "L" else s_r1
            t_keep_0 = t_l0 if keep == "L" else t_r0
            t_keep_1 = t_l1 if keep == "L" else t_r1
            t_keep_cw = t_l_cw if keep == "L" else t_r_cw

            if t0 == 1:
                s0 = _xor_bytes(s_keep_0, s_cw)
                t0 = t_keep_0 ^ t_keep_cw
            else:
                s0 = s_keep_0
                t0 = t_keep_0

            if t1 == 1:
                s1 = _xor_bytes(s_keep_1, s_cw)
                t1 = t_keep_1 ^ t_keep_cw
            else:
                s1 = s_keep_1
                t1 = t_keep_1

        final_cw = self.convert_gen(t1, s0, s1, beta)
        key0 = CoreFSSKey(seed0=s0_init, t0=t0_init, cws=tuple(cws), final_cw=final_cw)
        key1 = CoreFSSKey(seed0=s1_init, t0=t1_init, cws=tuple(cws), final_cw=final_cw)
        path_state = CorePathState(
            t_n_1=t1, s_n_0=s0, s_n_1=s1, cws=tuple(cws), n_bits=n_bits
        )
        return key0, key1, path_state

    # ---------- Algorithm 5: ConvertGen ----------
    def convert_gen(
        self,
        t_n_1: int,
        s_n_0: bytes,
        s_n_1: bytes,
        beta: Sequence[int],
    ) -> Tuple[int, ...]:
        temp = self.sub(list(beta), self.convert_g(s_n_0))
        temp = self.add(temp, self.convert_g(s_n_1))
        if t_n_1 == 1:
            temp = self.neg(temp)
        return tuple(temp)

    # ---------- Algorithm 6: PathEval ----------
    def path_eval(
        self,
        key: CoreFSSKey,
        x: int,
        n_bits: int | None = None,
    ) -> Tuple[int, bytes]:
        if n_bits is None:
            n_bits = len(key.cws)

        x_bits = self._int_to_bits(x, n_bits)
        s = key.seed0
        t = key.t0

        for bit, cw in zip(x_bits, key.cws):
            s_l_raw, t_l_raw, s_r_raw, t_r_raw = self.prg(s)
            if t == 1:
                s_l = _xor_bytes(s_l_raw, cw.s_cw)
                t_l = t_l_raw ^ cw.t_l_cw
                s_r = _xor_bytes(s_r_raw, cw.s_cw)
                t_r = t_r_raw ^ cw.t_r_cw
            else:
                s_l, t_l, s_r, t_r = s_l_raw, t_l_raw, s_r_raw, t_r_raw

            if bit == 0:
                s, t = s_l, t_l
            else:
                s, t = s_r, t_r

        return t, s

    # ---------- Algorithm 4: ConvertEval ----------
    def convert_eval(
        self,
        server_bit: int,
        t_n: int,
        s_n: bytes,
        final_cw: Sequence[int],
    ) -> Vec:
        out = self.add(self.convert_g(s_n), self.scalar_bit_mul(t_n, final_cw))
        if server_bit == 1:
            out = self.neg(out)
        return out

    def eval(
        self,
        server_bit: int,
        key: CoreFSSKey,
        x: int,
        n_bits: int | None = None,
    ) -> Vec:
        t_n, s_n = self.path_eval(key, x, n_bits=n_bits)
        return self.convert_eval(server_bit, t_n, s_n, key.final_cw)


# ---------------------------------------------------------------------------
# Torch adapter used by the optimized protocol
# ---------------------------------------------------------------------------


@dataclass
class CachedPathState:
    """
    Cached state for an index path.

    For the current prototype we only need the index `alpha` on the
    caller side; all FSS-specific state is kept inside the adapter.
    """

    alpha: int


@dataclass
class FSSKey:
    """
    Wrapper around a core SecEmb FSS key for one server.
    """

    alpha: int
    server_bit: int  # 0 or 1
    core_key: CoreFSSKey


@dataclass
class PartialKey:
    """
    Wrapper used for the sparse upload phase.
    """

    alpha: int
    server_bit: int  # 0 or 1
    core_key: CoreFSSKey


class PointFSS:
    """
    Torch-facing backend that adapts ``SecEmbFSS`` to the prototype API.

    Parameters
    ----------
    num_items:
        Size of the item index domain.
    embedding_dim:
        Dimension of item embedding / gradient rows.
    lambda_bits, modulus:
        Security parameter and group modulus, forwarded to ``SecEmbFSS``.
    scale:
        Fixed-point scale factor used to map floats to integers in Z_q.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        lambda_bits: int = 128,
        modulus: int = 2**61 - 1,
        scale: float = 1e6,
    ) -> None:
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.scale = float(scale)
        self.modulus = int(modulus)

        # Core FSS instances: scalar payload for retrieval, vector payload
        # for sparse row uploads.
        self._retrieval_fss = SecEmbFSS(
            lambda_bits=lambda_bits,
            modulus=self.modulus,
            dim=1,
        )
        self._agg_fss = SecEmbFSS(
            lambda_bits=lambda_bits,
            modulus=self.modulus,
            dim=self.embedding_dim,
        )

    # ------------------------------------------------------------------
    # Encoding helpers between torch tensors and Z_q vectors
    # ------------------------------------------------------------------

    def _encode_scalar(self, x: float) -> List[int]:
        v = int(round(float(x) * self.scale)) % self.modulus
        return [v]

    def _decode_scalar(self, vec: List[int]) -> float:
        if not vec:
            return 0.0
        return float(vec[0]) / self.scale

    def _encode_vec(self, beta: torch.Tensor) -> List[int]:
        flat = beta.detach().cpu().reshape(-1).tolist()
        if len(flat) != self.embedding_dim:
            raise ValueError(
                f"Expected beta of length {self.embedding_dim}, got {len(flat)}"
            )
        return [int(round(float(v) * self.scale)) % self.modulus for v in flat]

    def _decode_vec(self, vec: List[int]) -> torch.Tensor:
        if len(vec) != self.embedding_dim:
            raise ValueError(
                f"Expected vector of length {self.embedding_dim}, got {len(vec)}"
            )
        floats = [float(v) / self.scale for v in vec]
        return torch.tensor(floats, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public API used by the optimized protocol
    # ------------------------------------------------------------------

    def gen_with_cache(
        self,
        alpha: int,
        beta: torch.Tensor,
        rng,
    ) -> Tuple[FSSKey, FSSKey, CachedPathState]:
        """
        Generate retrieval keys and a cached path state for index ``alpha``.

        In the current optimized protocol, ``beta`` is a scalar 1.0 payload.
        """
        del rng  # randomness handled inside SecEmbFSS

        n_bits = self._retrieval_fss.bits_for_domain(self.num_items)
        if beta.numel() != 1:
            raise ValueError("retrieval beta must be a scalar tensor")
        beta_scalar = float(beta.item())
        beta_vec = self._encode_scalar(beta_scalar)

        k0_core, k1_core, _ = self._retrieval_fss.gen(
            alpha=int(alpha),
            beta=beta_vec,
            n_bits=n_bits,
        )

        k0 = FSSKey(alpha=int(alpha), server_bit=0, core_key=k0_core)
        k1 = FSSKey(alpha=int(alpha), server_bit=1, core_key=k1_core)
        cache = CachedPathState(alpha=int(alpha))
        return k0, k1, cache

    def eval(self, key: FSSKey, x: int) -> torch.Tensor:
        """
        Evaluate this party's share at index ``x`` for retrieval.

        Returns a scalar torch tensor that is multiplied with the item
        embedding in the protocol code.
        """
        n_bits = self._retrieval_fss.bits_for_domain(self.num_items)
        vec = self._retrieval_fss.eval(
            server_bit=key.server_bit,
            key=key.core_key,
            x=int(x),
            n_bits=n_bits,
        )
        coeff = self._decode_scalar(vec)
        return torch.tensor(coeff, dtype=torch.float32)

    # --- Optimized sparse upload path ---

    def convert_gen(
        self,
        cache: CachedPathState,
        beta: torch.Tensor,
        rng,
    ) -> Tuple[PartialKey, PartialKey]:
        """
        Generate new keys for a sparse row payload using the cached index.
        """
        del rng  # randomness handled inside SecEmbFSS

        n_bits = self._agg_fss.bits_for_domain(self.num_items)
        beta_vec = self._encode_vec(beta)

        k0_core, k1_core, _ = self._agg_fss.gen(
            alpha=int(cache.alpha),
            beta=beta_vec,
            n_bits=n_bits,
        )

        pk0 = PartialKey(alpha=int(cache.alpha), server_bit=0, core_key=k0_core)
        pk1 = PartialKey(alpha=int(cache.alpha), server_bit=1, core_key=k1_core)
        return pk0, pk1

    def convert_eval(
        self,
        cache: CachedPathState,
        partial_key: PartialKey,
        x: int,
    ) -> torch.Tensor:
        """
        Evaluate this party's share for the sparse upload phase.

        For a proper point-function FSS, the output is non-zero only at
        ``x == cache.alpha``; callers in the optimized protocol only
        evaluate at that point.
        """
        n_bits = self._agg_fss.bits_for_domain(self.num_items)
        vec = self._agg_fss.eval(
            server_bit=partial_key.server_bit,
            key=partial_key.core_key,
            x=int(x),
            n_bits=n_bits,
        )
        return self._decode_vec(vec)

