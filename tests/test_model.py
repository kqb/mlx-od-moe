"""
Tests for the full transformer model with OD-MoE layers.

Uses small configs for fast testing.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx_od_moe.model import (
    KimiODMoEConfig,
    KimiODMoEModel,
    KVCache,
    MLACache,
    Attention,
    MLAttention,
    DenseMLP,
    TransformerBlock,
    _create_causal_mask,
    _sample_top_p,
    _expand_kv_heads,
)


def small_config():
    """Small GQA config for fast testing (backward compatible)."""
    return KimiODMoEConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        kv_lora_rank=0,
        q_lora_rank=0,
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        v_head_dim=0,
        num_experts_per_tok=2,
        num_local_experts=8,
        n_shared_experts=0,
        first_k_dense_replace=0,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        max_position_embeddings=512,
    )


def small_mla_config():
    """Small MLA config for testing K2.5-style attention."""
    return KimiODMoEConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_nope_head_dim=24,
        qk_rope_head_dim=8,
        v_head_dim=24,
        num_experts_per_tok=2,
        num_local_experts=8,
        n_shared_experts=1,
        first_k_dense_replace=1,
        scoring_func="sigmoid",
        routed_scaling_factor=1.5,
        max_position_embeddings=512,
    )


# ===========================================================================
# Config tests
# ===========================================================================


class TestKimiODMoEConfig:
    def test_default_config(self):
        config = KimiODMoEConfig()
        assert config.vocab_size == 163840
        assert config.hidden_size == 7168
        assert config.num_hidden_layers == 61
        assert config.num_attention_heads == 64
        assert config.kv_lora_rank == 512
        assert config.q_lora_rank == 1536
        assert config.qk_nope_head_dim == 128
        assert config.qk_rope_head_dim == 64
        assert config.v_head_dim == 128
        assert config.head_dim == 192  # 128 + 64
        assert config.n_shared_experts == 1
        assert config.first_k_dense_replace == 1
        assert config.scoring_func == "sigmoid"
        assert config.routed_scaling_factor == 2.827
        assert config.eos_token_id == 0
        assert config.shadow_lookahead == 2

    def test_gqa_config(self):
        config = small_config()
        assert config.kv_lora_rank == 0
        assert config.head_dim == 32  # 128 / 4
        assert config.vocab_size == 1000

    def test_mla_config(self):
        config = small_mla_config()
        assert config.kv_lora_rank == 32
        assert config.q_lora_rank == 64
        assert config.head_dim == 32  # 24 + 8
        assert config.n_shared_experts == 1
        assert config.first_k_dense_replace == 1
        assert config.scoring_func == "sigmoid"

    def test_custom_eos_and_lookahead(self):
        config = KimiODMoEConfig(eos_token_id=2, shadow_lookahead=1)
        assert config.eos_token_id == 2
        assert config.shadow_lookahead == 1


# ===========================================================================
# KV Cache tests (GQA)
# ===========================================================================


class TestKVCache:
    def test_empty_cache(self):
        cache = KVCache()
        assert cache.offset == 0
        assert cache.keys is None

    def test_first_update(self):
        cache = KVCache()
        keys = mx.random.normal((1, 4, 8, 32))
        values = mx.random.normal((1, 4, 8, 32))
        k, v = cache.update(keys, values)
        assert cache.offset == 8
        assert k.shape == (1, 4, 8, 32)

    def test_incremental_update(self):
        cache = KVCache()
        k1 = mx.random.normal((1, 4, 8, 32))
        v1 = mx.random.normal((1, 4, 8, 32))
        cache.update(k1, v1)
        assert cache.offset == 8

        k2 = mx.random.normal((1, 4, 1, 32))
        v2 = mx.random.normal((1, 4, 1, 32))
        k, v = cache.update(k2, v2)
        assert cache.offset == 9
        assert k.shape == (1, 4, 9, 32)

    def test_multiple_updates(self):
        cache = KVCache()
        for i in range(5):
            k = mx.random.normal((1, 2, 1, 16))
            v = mx.random.normal((1, 2, 1, 16))
            cache.update(k, v)
        assert cache.offset == 5

    def test_preallocated_buffer_reuses_memory(self):
        cache = KVCache(step=8)
        k1 = mx.random.normal((1, 2, 3, 16))
        v1 = mx.random.normal((1, 2, 3, 16))
        cache.update(k1, v1)
        assert cache.keys.shape[2] >= 3
        buf_size_after_first = cache.keys.shape[2]

        for _ in range(4):
            k = mx.random.normal((1, 2, 1, 16))
            v = mx.random.normal((1, 2, 1, 16))
            cache.update(k, v)
        assert cache.offset == 7
        assert cache.keys.shape[2] == buf_size_after_first

    def test_values_preserved_across_growth(self):
        cache = KVCache(step=4)
        k1 = mx.ones((1, 1, 3, 4))
        v1 = mx.ones((1, 1, 3, 4)) * 2.0
        cache.update(k1, v1)
        mx.eval(cache.keys, cache.values)

        k2 = mx.ones((1, 1, 3, 4)) * 3.0
        v2 = mx.ones((1, 1, 3, 4)) * 4.0
        k_out, v_out = cache.update(k2, v2)
        mx.eval(k_out, v_out)
        assert k_out[0, 0, 0, 0].item() == 1.0
        assert v_out[0, 0, 0, 0].item() == 2.0
        assert k_out[0, 0, 3, 0].item() == 3.0
        assert v_out[0, 0, 3, 0].item() == 4.0


# ===========================================================================
# MLA Cache tests
# ===========================================================================


class TestMLACache:
    def test_empty_cache(self):
        cache = MLACache()
        assert cache.offset == 0
        assert cache.compressed_kv is None

    def test_first_update(self):
        cache = MLACache()
        ckv = mx.random.normal((1, 8, 32))  # (B, L, kv_lora_rank)
        kpe = mx.random.normal((1, 8, 8))   # (B, L, rope_dim)
        ckv_out, kpe_out = cache.update(ckv, kpe)
        assert cache.offset == 8
        assert ckv_out.shape == (1, 8, 32)
        assert kpe_out.shape == (1, 8, 8)

    def test_incremental_update(self):
        cache = MLACache()
        ckv1 = mx.random.normal((1, 8, 32))
        kpe1 = mx.random.normal((1, 8, 8))
        cache.update(ckv1, kpe1)
        assert cache.offset == 8

        ckv2 = mx.random.normal((1, 1, 32))
        kpe2 = mx.random.normal((1, 1, 8))
        ckv_out, kpe_out = cache.update(ckv2, kpe2)
        assert cache.offset == 9
        assert ckv_out.shape == (1, 9, 32)
        assert kpe_out.shape == (1, 9, 8)

    def test_values_preserved_across_growth(self):
        cache = MLACache(step=4)
        ckv1 = mx.ones((1, 3, 4))
        kpe1 = mx.ones((1, 3, 2)) * 2.0
        cache.update(ckv1, kpe1)
        mx.eval(cache.compressed_kv, cache.k_pe)

        ckv2 = mx.ones((1, 3, 4)) * 3.0
        kpe2 = mx.ones((1, 3, 2)) * 4.0
        ckv_out, kpe_out = cache.update(ckv2, kpe2)
        mx.eval(ckv_out, kpe_out)
        assert ckv_out[0, 0, 0].item() == 1.0
        assert kpe_out[0, 0, 0].item() == 2.0
        assert ckv_out[0, 3, 0].item() == 3.0
        assert kpe_out[0, 3, 0].item() == 4.0

    def test_preallocated_buffer_reuses(self):
        cache = MLACache(step=8)
        ckv = mx.random.normal((1, 3, 16))
        kpe = mx.random.normal((1, 3, 4))
        cache.update(ckv, kpe)
        buf_size = cache.compressed_kv.shape[1]

        for _ in range(4):
            cache.update(mx.random.normal((1, 1, 16)), mx.random.normal((1, 1, 4)))
        assert cache.offset == 7
        assert cache.compressed_kv.shape[1] == buf_size


# ===========================================================================
# Causal mask tests
# ===========================================================================


class TestCausalMask:
    def test_single_token_no_mask(self):
        mask = _create_causal_mask(1)
        assert mask is None

    def test_mask_shape(self):
        mask = _create_causal_mask(8)
        assert mask.shape == (8, 8)

    def test_mask_is_causal(self):
        mask = _create_causal_mask(4)
        mx.eval(mask)
        for i in range(4):
            for j in range(i + 1):
                assert mask[i, j].item() == 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j].item() < -1e8

    def test_mask_with_cache_offset(self):
        mask = _create_causal_mask(query_len=3, kv_len=8)
        mx.eval(mask)
        assert mask.shape == (3, 8)
        for j in range(6):
            assert mask[0, j].item() == 0.0
        for j in range(6, 8):
            assert mask[0, j].item() < -1e8
        for j in range(8):
            assert mask[2, j].item() == 0.0

    def test_mask_caching(self):
        m1 = _create_causal_mask(4, 4)
        m2 = _create_causal_mask(4, 4)
        assert m1 is m2


# ===========================================================================
# GQA Attention tests
# ===========================================================================


class TestExpandKVHeads:
    def test_no_expansion(self):
        x = mx.random.normal((1, 4, 8, 32))
        result = _expand_kv_heads(x, 1)
        assert result is x

    def test_expansion_shape(self):
        x = mx.random.normal((1, 2, 8, 32))
        result = _expand_kv_heads(x, 4)
        assert result.shape == (1, 8, 8, 32)

    def test_expansion_values_correct(self):
        x = mx.array([[[[1.0, 2.0]], [[3.0, 4.0]]]])
        result = _expand_kv_heads(x, 2)
        mx.eval(result)
        assert result.shape == (1, 4, 1, 2)
        assert result[0, 0, 0, 0].item() == 1.0
        assert result[0, 1, 0, 0].item() == 1.0
        assert result[0, 2, 0, 0].item() == 3.0
        assert result[0, 3, 0, 0].item() == 3.0


class TestAttention:
    def test_output_shape(self):
        config = small_config()
        attn = Attention(config)
        x = mx.random.normal((2, 8, 128))
        output = attn(x)
        assert output.shape == (2, 8, 128)

    def test_with_causal_mask(self):
        config = small_config()
        attn = Attention(config)
        x = mx.random.normal((1, 16, 128))
        mask = _create_causal_mask(16)
        output = attn(x, mask=mask)
        assert output.shape == (1, 16, 128)

    def test_with_kv_cache_prefill(self):
        config = small_config()
        attn = Attention(config)
        cache = KVCache()
        x = mx.random.normal((1, 8, 128))
        mask = _create_causal_mask(8)
        output = attn(x, mask=mask, cache=cache)
        assert output.shape == (1, 8, 128)
        assert cache.offset == 8

    def test_with_kv_cache_decode(self):
        config = small_config()
        attn = Attention(config)
        cache = KVCache()
        x1 = mx.random.normal((1, 8, 128))
        mask = _create_causal_mask(8)
        attn(x1, mask=mask, cache=cache)
        assert cache.offset == 8

        x2 = mx.random.normal((1, 1, 128))
        output = attn(x2, cache=cache)
        assert output.shape == (1, 1, 128)
        assert cache.offset == 9

    def test_gqa_head_expansion(self):
        config = small_config()
        attn = Attention(config)
        assert attn.num_kv_groups == 2
        x = mx.random.normal((1, 4, 128))
        output = attn(x)
        assert output.shape == (1, 4, 128)


# ===========================================================================
# MLA Attention tests
# ===========================================================================


class TestMLAttention:
    def test_output_shape(self):
        config = small_mla_config()
        attn = MLAttention(config)
        x = mx.random.normal((1, 8, 128))
        output = attn(x)
        mx.eval(output)
        assert output.shape == (1, 8, 128)

    def test_output_not_nan(self):
        config = small_mla_config()
        attn = MLAttention(config)
        x = mx.random.normal((1, 4, 128))
        output = attn(x)
        mx.eval(output)
        assert not mx.any(mx.isnan(output))

    def test_with_causal_mask(self):
        config = small_mla_config()
        attn = MLAttention(config)
        x = mx.random.normal((1, 16, 128))
        mask = _create_causal_mask(16)
        output = attn(x, mask=mask)
        mx.eval(output)
        assert output.shape == (1, 16, 128)

    def test_with_cache_prefill(self):
        config = small_mla_config()
        attn = MLAttention(config)
        cache = MLACache()
        x = mx.random.normal((1, 8, 128))
        mask = _create_causal_mask(8)
        output = attn(x, mask=mask, cache=cache)
        mx.eval(output)
        assert output.shape == (1, 8, 128)
        assert cache.offset == 8

    def test_with_cache_decode(self):
        config = small_mla_config()
        attn = MLAttention(config)
        cache = MLACache()
        x1 = mx.random.normal((1, 8, 128))
        mask = _create_causal_mask(8)
        attn(x1, mask=mask, cache=cache)
        mx.eval(cache.compressed_kv)
        assert cache.offset == 8

        x2 = mx.random.normal((1, 1, 128))
        output = attn(x2, cache=cache)
        mx.eval(output)
        assert output.shape == (1, 1, 128)
        assert cache.offset == 9

    def test_batch(self):
        config = small_mla_config()
        attn = MLAttention(config)
        x = mx.random.normal((2, 4, 128))
        output = attn(x)
        mx.eval(output)
        assert output.shape == (2, 4, 128)

    def test_without_q_lora(self):
        """Test MLA with direct query projection (q_lora_rank=0)."""
        config = small_mla_config()
        config.q_lora_rank = 0
        attn = MLAttention(config)
        x = mx.random.normal((1, 4, 128))
        output = attn(x)
        mx.eval(output)
        assert output.shape == (1, 4, 128)


# ===========================================================================
# DenseMLP tests
# ===========================================================================


class TestDenseMLP:
    def test_output_shape(self):
        config = small_mla_config()
        mlp = DenseMLP(config)
        x = mx.random.normal((1, 8, 128))
        output = mlp(x)
        mx.eval(output)
        assert output.shape == (1, 8, 128)

    def test_not_nan(self):
        config = small_mla_config()
        mlp = DenseMLP(config)
        x = mx.random.normal((1, 4, 128))
        output = mlp(x)
        mx.eval(output)
        assert not mx.any(mx.isnan(output))


# ===========================================================================
# TransformerBlock tests
# ===========================================================================


class TestTransformerBlock:
    def test_gqa_output_shape(self):
        config = small_config()
        block = TransformerBlock(config, layer_idx=0)
        x = mx.random.normal((1, 8, 128))
        output = block(x)
        assert output.shape == (1, 8, 128)

    def test_gqa_with_mask_and_cache(self):
        config = small_config()
        block = TransformerBlock(config, layer_idx=0)
        cache = KVCache()
        x = mx.random.normal((1, 8, 128))
        mask = _create_causal_mask(8)
        output = block(x, mask=mask, cache=cache)
        assert output.shape == (1, 8, 128)
        assert cache.offset == 8

    def test_residual_connection(self):
        config = small_config()
        block = TransformerBlock(config, layer_idx=0)
        assert block.ffn is None
        x = mx.random.normal((1, 4, 128))
        output = block(x)
        mx.eval(output)
        assert not mx.allclose(output, x, atol=1e-6)

    def test_with_moe_layer(self):
        from mlx_od_moe.od_moe_layer import ODMoELayer
        config = small_config()
        block = TransformerBlock(config, layer_idx=0)
        block.ffn = ODMoELayer(
            layer_idx=0, hidden_dim=128, ffn_dim=256, num_experts=8, top_k=2,
        )
        x = mx.random.normal((1, 4, 128))
        output = block(x)
        assert output.shape == (1, 4, 128)

    def test_mla_block_output_shape(self):
        config = small_mla_config()
        block = TransformerBlock(config, layer_idx=0)
        assert isinstance(block.attention, MLAttention)
        assert isinstance(block.ffn, DenseMLP)  # layer 0 < first_k_dense_replace=1
        x = mx.random.normal((1, 4, 128))
        output = block(x)
        mx.eval(output)
        assert output.shape == (1, 4, 128)

    def test_mla_block_moe_layer(self):
        """MoE layers (idx >= first_k_dense_replace) start with ffn=None."""
        config = small_mla_config()
        block = TransformerBlock(config, layer_idx=1)
        assert block.ffn is None  # needs setup_od_moe

    def test_mla_block_with_cache(self):
        config = small_mla_config()
        block = TransformerBlock(config, layer_idx=0)
        cache = MLACache()
        x = mx.random.normal((1, 8, 128))
        mask = _create_causal_mask(8)
        output = block(x, mask=mask, cache=cache)
        mx.eval(output)
        assert output.shape == (1, 8, 128)
        assert cache.offset == 8


# ===========================================================================
# Full model tests (GQA mode)
# ===========================================================================


class TestKimiODMoEModel:
    def test_forward_pass_shape(self):
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = model(input_ids)
        assert logits.shape == (1, 5, 1000)

    def test_forward_batch(self):
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3], [4, 5, 6]])
        logits = model(input_ids)
        assert logits.shape == (2, 3, 1000)

    def test_forward_with_cache(self):
        config = small_config()
        model = KimiODMoEModel(config)
        cache = [KVCache() for _ in model.layers]
        input_ids = mx.array([[1, 2, 3, 4]])
        logits = model(input_ids, cache=cache)
        assert logits.shape == (1, 4, 1000)
        assert cache[0].offset == 4

        input_ids = mx.array([[5]])
        logits = model(input_ids, cache=cache)
        assert logits.shape == (1, 1, 1000)
        assert cache[0].offset == 5

    def test_logits_not_nan(self):
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3]])
        logits = model(input_ids)
        mx.eval(logits)
        assert not mx.any(mx.isnan(logits))

    def test_layer_count(self):
        config = small_config()
        model = KimiODMoEModel(config)
        assert len(model.layers) == 2

    def test_generate_produces_tokens(self):
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3]])
        tokens = list(model.generate(input_ids, max_new_tokens=5, temperature=0.8))
        assert len(tokens) > 0
        assert len(tokens) <= 5
        for t in tokens:
            assert isinstance(t, int)
            assert 0 <= t < config.vocab_size

    def test_generate_greedy(self):
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3]])
        tokens1 = list(model.generate(input_ids, max_new_tokens=3, temperature=0))
        tokens2 = list(model.generate(input_ids, max_new_tokens=3, temperature=0))
        assert tokens1 == tokens2

    def test_generate_stops_on_default_eos(self):
        config = small_config()
        model = KimiODMoEModel(config)
        model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
        input_ids = mx.array([[1, 2]])
        tokens = list(model.generate(input_ids, max_new_tokens=100, temperature=0))
        assert len(tokens) == 1
        assert tokens[0] == 0

    def test_generate_custom_eos(self):
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2]])
        tokens_no_eos = list(
            model.generate(input_ids, max_new_tokens=5, temperature=0, eos_token_id=-1)
        )
        first_token = tokens_no_eos[0]
        assert len(tokens_no_eos) == 5

        tokens_with_eos = list(
            model.generate(
                input_ids, max_new_tokens=100, temperature=0, eos_token_id=first_token
            )
        )
        assert len(tokens_with_eos) == 1
        assert tokens_with_eos[0] == first_token

    def test_generate_log_interval(self):
        config = small_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3]])
        list(model.generate(input_ids, max_new_tokens=3, temperature=0.5, log_interval=0))
        list(model.generate(input_ids, max_new_tokens=3, temperature=0.5, log_interval=1))


# ===========================================================================
# Full model tests (MLA mode)
# ===========================================================================


class TestKimiODMoEModelMLA:
    def test_forward_pass_shape(self):
        config = small_mla_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = model(input_ids)
        mx.eval(logits)
        assert logits.shape == (1, 5, 1000)

    def test_forward_not_nan(self):
        config = small_mla_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3]])
        logits = model(input_ids)
        mx.eval(logits)
        assert not mx.any(mx.isnan(logits))

    def test_forward_with_cache(self):
        config = small_mla_config()
        model = KimiODMoEModel(config)
        cache = model._make_cache()
        assert isinstance(cache[0], MLACache)

        input_ids = mx.array([[1, 2, 3, 4]])
        logits = model(input_ids, cache=cache)
        mx.eval(logits)
        assert logits.shape == (1, 4, 1000)
        assert cache[0].offset == 4

        input_ids = mx.array([[5]])
        logits = model(input_ids, cache=cache)
        mx.eval(logits)
        assert logits.shape == (1, 1, 1000)
        assert cache[0].offset == 5

    def test_generate_tokens(self):
        config = small_mla_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3]])
        tokens = list(model.generate(input_ids, max_new_tokens=5, temperature=0.8))
        assert len(tokens) > 0
        assert len(tokens) <= 5

    def test_generate_greedy_deterministic(self):
        config = small_mla_config()
        model = KimiODMoEModel(config)
        input_ids = mx.array([[1, 2, 3]])
        tokens1 = list(model.generate(input_ids, max_new_tokens=3, temperature=0))
        tokens2 = list(model.generate(input_ids, max_new_tokens=3, temperature=0))
        assert tokens1 == tokens2

    def test_dense_plus_moe_layers(self):
        """First layer should be dense, second should start as None."""
        config = small_mla_config()
        model = KimiODMoEModel(config)
        assert isinstance(model.layers[0].ffn, DenseMLP)
        assert model.layers[1].ffn is None


# ===========================================================================
# Shared expert and sigmoid routing tests
# ===========================================================================


class TestODMoELayerExtensions:
    def test_shared_expert_output(self):
        from mlx_od_moe.od_moe_layer import ODMoELayer
        layer = ODMoELayer(
            layer_idx=0, hidden_dim=128, ffn_dim=64, num_experts=8, top_k=2,
            n_shared_experts=1,
        )
        x = mx.random.normal((1, 4, 128))
        output = layer(x)
        mx.eval(output)
        assert output.shape == (1, 4, 128)
        assert not mx.any(mx.isnan(output))

    def test_sigmoid_routing(self):
        from mlx_od_moe.od_moe_layer import ODMoELayer
        layer = ODMoELayer(
            layer_idx=0, hidden_dim=128, ffn_dim=64, num_experts=8, top_k=2,
            scoring_func="sigmoid", routed_scaling_factor=2.0,
        )
        x = mx.random.normal((1, 4, 128))
        output = layer(x)
        mx.eval(output)
        assert output.shape == (1, 4, 128)

    def test_shared_expert_and_sigmoid(self):
        from mlx_od_moe.od_moe_layer import ODMoELayer
        layer = ODMoELayer(
            layer_idx=0, hidden_dim=128, ffn_dim=64, num_experts=8, top_k=2,
            n_shared_experts=1, scoring_func="sigmoid",
            routed_scaling_factor=1.5, num_layers=4,
        )
        x = mx.random.normal((1, 4, 128))
        output = layer(x)
        mx.eval(output)
        assert output.shape == (1, 4, 128)

    def test_no_shared_expert_backward_compat(self):
        from mlx_od_moe.od_moe_layer import ODMoELayer
        layer = ODMoELayer(
            layer_idx=0, hidden_dim=128, ffn_dim=256, num_experts=8, top_k=2,
        )
        assert layer.shared_gate_proj is None
        x = mx.random.normal((1, 4, 128))
        output = layer(x)
        mx.eval(output)
        assert output.shape == (1, 4, 128)


# ===========================================================================
# Top-p sampling tests
# ===========================================================================


class TestSampleTopP:
    def test_basic_sampling(self):
        logits = mx.array([[0.0, 1.0, 2.0, 3.0]])
        token = _sample_top_p(logits, p=0.9)
        mx.eval(token)
        assert 0 <= token.item() < 4

    def test_p_1_uses_all_tokens(self):
        logits = mx.array([[1.0, 1.0, 1.0, 1.0]])
        token = _sample_top_p(logits, p=1.0)
        mx.eval(token)
        assert 0 <= token.item() < 4

    def test_low_p_concentrates(self):
        logits = mx.array([[-100.0, -100.0, -100.0, 100.0]])
        samples = []
        for _ in range(10):
            token = _sample_top_p(logits, p=0.1)
            mx.eval(token)
            samples.append(token.item())
        assert all(s == 3 for s in samples)

    def test_batch_sampling(self):
        logits = mx.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
        tokens = _sample_top_p(logits, p=0.9)
        mx.eval(tokens)
        assert tokens.shape == (2,)
