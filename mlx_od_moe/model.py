"""
Full Model - Transformer with OD-MoE layers

Implements the complete Kimi-K2.5 inference pipeline:
- MLA (Multi-head Latent Attention) with decoupled RoPE and absorb path
- GQA (Grouped Query Attention) fallback for other architectures
- Pre-allocated KV cache (standard and compressed MLA variants)
- OD-MoE FFN layers with on-demand expert loading
- Dense FFN for initial layers (first_k_dense_replace)
- Top-p (nucleus) sampling
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Generator
from pathlib import Path

from .expert_store import UnifiedMemoryExpertStore
from .shadow_model import ShadowRunner
from .od_moe_layer import ODMoELayer


class KimiODMoEConfig:
    """Configuration matching Kimi-K2.5 architecture."""

    def __init__(
        self,
        vocab_size=163840,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        num_hidden_layers=61,
        num_attention_heads=64,
        num_key_value_heads=8,
        # MLA parameters (set kv_lora_rank=0 for GQA fallback)
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        # MoE parameters
        num_experts_per_tok=8,
        num_local_experts=384,
        n_shared_experts=1,
        first_k_dense_replace=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.827,
        # Position encoding
        rope_theta=50000.0,
        max_position_embeddings=262144,
        # Other
        rms_norm_eps=1e-6,
        eos_token_id=0,
        shadow_lookahead=2,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.n_shared_experts = n_shared_experts
        self.first_k_dense_replace = first_k_dense_replace
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.eos_token_id = eos_token_id
        self.shadow_lookahead = shadow_lookahead
        # Derived
        if kv_lora_rank > 0:
            self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        else:
            self.head_dim = hidden_size // num_attention_heads


class KVCache:
    """
    Pre-allocated KV cache for autoregressive generation.

    Allocates buffers in chunks (default 256 tokens) and fills them via
    slice assignment, avoiding O(T) copies on every decode step.
    """

    def __init__(self, step: int = 256):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self._offset = 0
        self._step = step

    @property
    def offset(self) -> int:
        return self._offset

    def update(self, keys: mx.array, values: mx.array):
        new_tokens = keys.shape[2]

        if self.keys is None or (self._offset + new_tokens) > self.keys.shape[2]:
            # Need to allocate or grow the buffer
            B, H, _, D = keys.shape
            n_steps = (self._step + new_tokens - 1) // self._step
            alloc_len = n_steps * self._step
            new_k = mx.zeros((B, H, self._offset + alloc_len, D), dtype=keys.dtype)
            new_v = mx.zeros((B, H, self._offset + alloc_len, D), dtype=values.dtype)

            if self.keys is not None:
                # Copy existing cached data into new buffer
                new_k[:, :, : self._offset, :] = self.keys[:, :, : self._offset, :]
                new_v[:, :, : self._offset, :] = self.values[:, :, : self._offset, :]

            self.keys = new_k
            self.values = new_v

        # Write new keys/values into the pre-allocated slot
        self.keys[:, :, self._offset : self._offset + new_tokens, :] = keys
        self.values[:, :, self._offset : self._offset + new_tokens, :] = values
        self._offset += new_tokens

        return self.keys[:, :, : self._offset, :], self.values[:, :, : self._offset, :]


class MLACache:
    """
    Pre-allocated cache for MLA compressed KV representations.

    Stores only the compressed latent (kv_lora_rank) and decoupled RoPE key
    (qk_rope_head_dim) per token per layer, giving ~28x reduction vs full KV.
    """

    def __init__(self, step: int = 256):
        self.compressed_kv: Optional[mx.array] = None  # (B, T, kv_lora_rank)
        self.k_pe: Optional[mx.array] = None           # (B, T, qk_rope_head_dim)
        self._offset = 0
        self._step = step

    @property
    def offset(self) -> int:
        return self._offset

    def update(self, compressed_kv: mx.array, k_pe: mx.array):
        """
        Update cache with new compressed KV and RoPE keys.

        Args:
            compressed_kv: (B, L, kv_lora_rank)
            k_pe: (B, L, qk_rope_head_dim)

        Returns:
            Tuple of (compressed_kv, k_pe) sliced to current length
        """
        new_tokens = compressed_kv.shape[1]

        if self.compressed_kv is None or (self._offset + new_tokens) > self.compressed_kv.shape[1]:
            B = compressed_kv.shape[0]
            kv_dim = compressed_kv.shape[2]
            pe_dim = k_pe.shape[2]

            n_steps = (self._step + new_tokens - 1) // self._step
            alloc_len = n_steps * self._step

            new_ckv = mx.zeros((B, self._offset + alloc_len, kv_dim), dtype=compressed_kv.dtype)
            new_kpe = mx.zeros((B, self._offset + alloc_len, pe_dim), dtype=k_pe.dtype)

            if self.compressed_kv is not None:
                new_ckv[:, :self._offset, :] = self.compressed_kv[:, :self._offset, :]
                new_kpe[:, :self._offset, :] = self.k_pe[:, :self._offset, :]

            self.compressed_kv = new_ckv
            self.k_pe = new_kpe

        self.compressed_kv[:, self._offset:self._offset + new_tokens, :] = compressed_kv
        self.k_pe[:, self._offset:self._offset + new_tokens, :] = k_pe
        self._offset += new_tokens

        return (
            self.compressed_kv[:, :self._offset, :],
            self.k_pe[:, :self._offset, :],
        )


def _expand_kv_heads(x: mx.array, n_rep: int) -> mx.array:
    """Expand KV heads to match query heads via zero-copy broadcast."""
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return mx.broadcast_to(
        x[:, :, None, :, :], (B, H, n_rep, T, D)
    ).reshape(B, H * n_rep, T, D)


# Cache for commonly used causal masks to avoid recomputation
_mask_cache: dict = {}


def _create_causal_mask(
    query_len: int, kv_len: Optional[int] = None
) -> Optional[mx.array]:
    """
    Create additive causal attention mask.

    Args:
        query_len: Number of query tokens
        kv_len: Total KV length (query_len + cache offset). If None, equals query_len.

    Returns:
        Mask of shape (query_len, kv_len) or None if no masking needed.
    """
    if query_len <= 1:
        return None

    if kv_len is None:
        kv_len = query_len

    cache_key = (query_len, kv_len)
    if cache_key in _mask_cache:
        return _mask_cache[cache_key]

    # Row indices: positions in the query (offset by kv_len - query_len)
    q_positions = mx.arange(query_len) + (kv_len - query_len)
    k_positions = mx.arange(kv_len)
    mask = (q_positions[:, None] < k_positions[None, :]).astype(mx.float32) * -1e9

    # Cache masks for common sizes (limit cache to avoid unbounded growth)
    if len(_mask_cache) < 64:
        _mask_cache[cache_key] = mask

    return mask


def _sample_top_p(logits: mx.array, p: float) -> mx.array:
    """
    Top-p (nucleus) sampling.

    Args:
        logits: Un-normalized logits (batch, vocab_size)
        p: Cumulative probability threshold

    Returns:
        Sampled token IDs (batch,)
    """
    if p >= 1.0:
        return mx.random.categorical(logits)

    probs = mx.softmax(logits, axis=-1)
    sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Mask tokens beyond the cumulative threshold (keep at least one)
    mask = (cumulative_probs - sorted_probs) >= p
    sorted_probs = mx.where(mask, mx.zeros_like(sorted_probs), sorted_probs)

    # Re-normalize
    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)

    # Sample: use -inf for zeroed-out tokens to avoid epsilon-induced distribution shift
    log_probs = mx.where(
        sorted_probs > 0,
        mx.log(sorted_probs),
        mx.full(sorted_probs.shape, -1e9),
    )
    sampled_idx = mx.random.categorical(log_probs)

    # Map back to original vocabulary indices
    next_token = mx.take_along_axis(sorted_indices, sampled_idx[:, None], axis=-1)[:, 0]
    return next_token


class Attention(nn.Module):
    """Grouped Query Attention with RoPE."""

    def __init__(self, config: KimiODMoEConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rope = nn.RoPE(self.head_dim, base=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape: (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim)
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE with cache offset
        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        # Update KV cache
        if cache is not None:
            keys, values = cache.update(keys, values)

        # GQA: expand KV heads to match query heads (zero-copy broadcast)
        keys = _expand_kv_heads(keys, self.num_kv_groups)
        values = _expand_kv_heads(values, self.num_kv_groups)

        # Scaled dot-product attention
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        output = weights @ values

        # Reshape back: (B, num_heads, L, head_dim) -> (B, L, hidden_size)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLAttention(nn.Module):
    """
    Multi-head Latent Attention with absorb path.

    Compresses KV into a low-rank latent space with decoupled RoPE.
    The absorb path moves W_UK/W_UV multiplication to query/output side,
    so the KV cache stores only compressed representations.

    Per-token cache: kv_lora_rank + qk_rope_head_dim values
    (e.g., 512 + 64 = 576 for K2.5, vs 16384 for standard MHA)
    """

    def __init__(self, config: KimiODMoEConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.scale = self.q_head_dim ** -0.5

        # Query path
        self.q_lora_rank = config.q_lora_rank
        if config.q_lora_rank > 0:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
            self.q_a_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.n_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                config.hidden_size, self.n_heads * self.q_head_dim, bias=False
            )

        # KV path (joint compressed + decoupled RoPE key)
        self.kv_a_proj = nn.Linear(
            config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=False
        )
        self.kv_a_norm = nn.RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.n_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )

        # Output
        self.o_proj = nn.Linear(
            self.n_heads * config.v_head_dim, config.hidden_size, bias=False
        )

        # RoPE for positional dimensions only
        self.rope = nn.RoPE(config.qk_rope_head_dim, base=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[MLACache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        # --- Query path ---
        if self.q_lora_rank > 0:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)

        q = q.reshape(B, L, self.n_heads, self.q_head_dim)
        q_nope = q[:, :, :, :self.qk_nope_head_dim]    # (B, L, n_h, nope_dim)
        q_pe = q[:, :, :, self.qk_nope_head_dim:]       # (B, L, n_h, rope_dim)

        # Transpose to (B, n_h, L, dim) for attention
        q_nope = q_nope.transpose(0, 2, 1, 3)
        q_pe = q_pe.transpose(0, 2, 1, 3)

        # --- KV path ---
        kv = self.kv_a_proj(x)                           # (B, L, kv_lora_rank + rope_dim)
        compressed_kv = kv[:, :, :self.kv_lora_rank]     # (B, L, kv_lora_rank)
        k_pe = kv[:, :, self.kv_lora_rank:]              # (B, L, rope_dim)
        compressed_kv = self.kv_a_norm(compressed_kv)

        # Apply RoPE to positional parts
        offset = cache.offset if cache is not None else 0

        # k_pe is shared across heads: (B, L, rope_dim) -> 4D for RoPE
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        k_pe = self.rope(k_pe, offset=offset)            # (B, 1, L, rope_dim)
        k_pe = k_pe.transpose(0, 2, 1, 3).reshape(B, L, self.qk_rope_head_dim)

        q_pe = self.rope(q_pe, offset=offset)            # (B, n_h, L, rope_dim)

        # --- Cache update ---
        if cache is not None:
            compressed_kv, k_pe = cache.update(compressed_kv, k_pe)

        # --- Absorb path attention ---
        # Extract W_UK and W_UV from kv_b_proj weight
        # weight shape: (n_h * (nope_dim + v_dim), kv_lora_rank)
        wkv_b = self.kv_b_proj.weight.reshape(
            self.n_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        W_UK = wkv_b[:, :self.qk_nope_head_dim, :]      # (n_h, nope_dim, kv_lora_rank)
        W_UV = wkv_b[:, self.qk_nope_head_dim:, :]      # (n_h, v_dim, kv_lora_rank)

        # Absorb W_UK into query: project q_nope into compressed space
        # (B, n_h, L, nope_dim) @ (n_h, nope_dim, kv_lora_rank) -> (B, n_h, L, kv_lora_rank)
        q_compressed = q_nope @ W_UK

        # Content scores: q_compressed @ compressed_kv^T
        # compressed_kv: (B, T, kv_lora_rank) -> (B, 1, kv_lora_rank, T)
        ckv_t = compressed_kv.transpose(0, 2, 1)[:, None, :, :]
        scores_nope = q_compressed @ ckv_t               # (B, n_h, L, T)

        # Position scores: q_pe @ k_pe^T
        # k_pe: (B, T, rope_dim) -> (B, 1, rope_dim, T)
        kpe_t = k_pe.transpose(0, 2, 1)[:, None, :, :]
        scores_pe = q_pe @ kpe_t                         # (B, n_h, L, T)

        # Combined scores
        scores = (scores_nope + scores_pe) * self.scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)

        # Weighted sum in compressed space
        # (B, n_h, L, T) @ (B, 1, T, kv_lora_rank) -> (B, n_h, L, kv_lora_rank)
        ckv_expanded = compressed_kv[:, None, :, :]
        attn_compressed = weights @ ckv_expanded

        # Decompress with W_UV^T
        # (B, n_h, L, kv_lora_rank) @ (n_h, kv_lora_rank, v_dim) -> (B, n_h, L, v_dim)
        W_UV_t = W_UV.transpose(0, 2, 1)
        output = attn_compressed @ W_UV_t

        # Reshape and project
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DenseMLP(nn.Module):
    """SwiGLU MLP for dense layers (non-MoE, used in first_k_dense_replace layers)."""

    def __init__(self, config: KimiODMoEConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        return self.down_proj(gate * mx.sigmoid(gate) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: Attention (MLA or GQA) + FFN (Dense or OD-MoE)."""

    def __init__(self, config: KimiODMoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # MLA or GQA based on config
        if config.kv_lora_rank > 0:
            self.attention = MLAttention(config)
        else:
            self.attention = Attention(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Dense FFN for first layers, MoE set up later via setup_od_moe
        if layer_idx < config.first_k_dense_replace:
            self.ffn = DenseMLP(config)
        else:
            self.ffn = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        # Pre-norm attention with residual
        r = self.attention(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r

        # Pre-norm FFN with residual (Dense or MoE)
        if self.ffn is not None:
            r = self.ffn(self.post_attention_layernorm(h))
            h = h + r

        return h


class KimiODMoEModel(nn.Module):
    """
    Full Kimi-K2.5 model with MLA attention and OD-MoE layers.

    Memory breakdown at inference (Q2, M3 Ultra 512GB):
    - Full model in memory: ~375GB
    - KV cache 256K (MLA compressed): ~15GB
    - OS + overhead: ~10GB
    - Total: ~400GB, ~112GB headroom

    With OD-MoE on M4 Max 36GB:
    - Base model (non-expert, Q4): ~9GB
    - Active experts (8 x 61 layers): ~10.7GB
    - KV cache 32K (MLA): ~2GB
    - Total: ~24GB resident + experts on NVMe
    """

    def __init__(self, config: KimiODMoEConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # OD-MoE components (initialized later)
        self.expert_store: Optional[UnifiedMemoryExpertStore] = None
        self.shadow_runner: Optional[ShadowRunner] = None

    def setup_od_moe(
        self,
        expert_dir: str,
        predictor_path: Optional[str] = None,
        cache_size_gb: int = 48,
    ):
        """Initialize OD-MoE after base model weights are loaded."""
        print("Setting up OD-MoE...")

        self.expert_store = UnifiedMemoryExpertStore(
            expert_dir,
            cache_size_gb=cache_size_gb,
            num_layers=self.config.num_hidden_layers,
            num_experts_per_layer=self.config.num_local_experts,
        )

        self.shadow_runner = ShadowRunner(predictor_path)

        for layer in self.layers:
            # Skip dense layers (they already have a DenseMLP ffn)
            if layer.layer_idx < self.config.first_k_dense_replace:
                continue
            layer.ffn = ODMoELayer(
                layer_idx=layer.layer_idx,
                hidden_dim=self.config.hidden_size,
                ffn_dim=self.config.moe_intermediate_size,
                num_experts=self.config.num_local_experts,
                top_k=self.config.num_experts_per_tok,
                expert_store=self.expert_store,
                shadow_runner=self.shadow_runner,
                n_shared_experts=self.config.n_shared_experts,
                scoring_func=self.config.scoring_func,
                routed_scaling_factor=self.config.routed_scaling_factor,
                num_layers=self.config.num_hidden_layers,
            )

        n_dense = self.config.first_k_dense_replace
        n_moe = len(self.layers) - n_dense
        print(f"OD-MoE setup complete: {n_dense} dense + {n_moe} MoE layers")

    def _make_cache(self) -> list:
        """Create appropriate cache list for the model's attention type."""
        if self.config.kv_lora_rank > 0:
            return [MLACache() for _ in self.layers]
        else:
            return [KVCache() for _ in self.layers]

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[list] = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            cache: Optional list of KVCache or MLACache per layer

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        h = self.embed_tokens(input_ids)

        # Build causal mask accounting for cached tokens
        cache_offset = cache[0].offset if cache else 0
        kv_len = h.shape[1] + cache_offset
        mask = _create_causal_mask(h.shape[1], kv_len)

        lookahead = self.config.shadow_lookahead
        eval_interval = max(1, self.config.num_hidden_layers // 4)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            h = layer(h, mask=mask, cache=layer_cache)

            # Trigger shadow model predictions for prefetch (configurable lookahead)
            if self.shadow_runner and i < self.config.num_hidden_layers - lookahead:
                self.shadow_runner.predict_async(h, i)

            # Periodic evaluation to bound graph memory on large models
            if i > 0 and i % eval_interval == 0:
                mx.eval(h)

        h = self.norm(h)
        return self.lm_head(h)

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.6,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        log_interval: int = 0,
    ) -> Generator[int, None, None]:
        """
        Streaming autoregressive generation with KV cache.

        Args:
            input_ids: Prompt token IDs (1, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            eos_token_id: Stop token (defaults to config.eos_token_id)
            log_interval: Print cache stats every N steps (0 = disabled)

        Yields:
            Generated token IDs one at a time
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        cache = self._make_cache()

        # Prefill: process full prompt
        logits = self(input_ids, cache=cache)
        mx.eval(logits)

        for step in range(max_new_tokens):
            next_token_logits = logits[:, -1, :]

            if temperature == 0:
                next_token = mx.argmax(next_token_logits, axis=-1)
            else:
                next_token = _sample_top_p(next_token_logits / temperature, top_p)
            mx.eval(next_token)

            token_id = next_token.item()
            yield token_id

            if token_id == eos_token_id:
                break

            # Decode step: process just the new token with KV cache
            logits = self(next_token.reshape(1, 1), cache=cache)
            mx.eval(logits)

            if log_interval > 0 and step % log_interval == 0 and self.expert_store:
                stats = self.expert_store.get_stats()
                print(f"Step {step}: Cache hit rate {stats['hit_rate']:.2%}")
