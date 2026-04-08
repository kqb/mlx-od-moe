# mlx-od-moe

On-Demand Mixture of Experts for Apple Silicon. Run 375GB models in 192GB RAM via memory-mapped expert loading.

## Overview

mlx-od-moe runs large Mixture-of-Experts (MoE) language models on Apple Silicon Macs by loading experts on-demand from disk instead of keeping them all in RAM.

It exploits Apple Silicon's unified memory plus fast NVMe to treat memory-mapped experts like an L3 cache: hot experts stay resident, cold experts are paged in from disk, and a lightweight shadow model predicts which experts to prefetch next.

## Why

Modern MoE models like Kimi-K2.5 (375GB) do not fit in RAM on a single consumer machine, even a 192GB Mac Studio. The usual workarounds are lossy (aggressive quantization), slow (CPU offload), or complex (multi-machine distributed inference).

mlx-od-moe keeps the model on disk and loads only the experts needed per token, with async prefetch driven by a learned predictor.

## Approach

1. Memory-map experts as `.npy` files on NVMe.
2. LRU cache of hot experts in RAM.
3. Shadow model predicts next top-K experts from hidden states.
4. Prefetch queue loads predicted experts asynchronously.

## Quick start

### Install

    git clone https://github.com/kqb/mlx-od-moe.git
    cd mlx-od-moe
    uv pip install -e .

Requirements:

- Python 3.11+
- macOS 14+ on Apple Silicon
- MLX
- NVMe storage with enough free space for the model's experts

### Convert a GGUF model

    python -m mlx_od_moe.convert \
      --input models/Kimi-K2.5.gguf \
      --output /Volumes/Storage/experts

### Run the inference server

    python -m mlx_od_moe.server \
      --expert-dir /Volumes/Storage/experts \
      --base-weights /Volumes/Storage/base_model.safetensors \
      --port 8080

### Query

    curl -X POST http://localhost:8080/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Explain quantum computing", "max_tokens": 512}'

## Architecture

    Model Layer
      Attention -> Router -> Experts (on-demand)

    On-Demand MoE Layer
      Shadow Model -> Prefetcher -> Expert Store (LRU cache)

    Memory-Mapped Expert Storage
      Expert 0, Expert 1, ... Expert N  (mmap .npy)

    NVMe SSD

### Components

- `expert_store.py`: LRU cache plus memory-mapped storage.
- `shadow_model.py`: Lightweight predictor for next top-K experts.
- `od_moe_layer.py`: Drop-in replacement for a standard MoE layer.
- `convert/`: GGUF extraction into memory-mappable NumPy arrays.
- `server.py`: HTTP inference server with metrics endpoint.

## Supported models

- Kimi-K2.5 (375GB total)
- Qwen2-57B-A14B
- Mixtral-8x7B
- Mixtral-8x22B

Planned: DeepSeek-V3, DBRX, custom MoE configs.

## Benchmarks

Benchmarks are workload and hardware dependent. Scripts to reproduce on your own machine are in `tests/`:

    python tests/benchmark_expert_fetch.py
    python tests/benchmark_inference.py
    python tests/benchmark_cache_hit_rate.py

Numbers from your own runs should go here rather than ones from someone else's box.

## Tuning

- Cache size: scale `max_cache_size` with free RAM after base weights.
- Storage: internal NVMe is best, Thunderbolt 4 NVMe is fine, USB 3.0 and network storage are not.
- Shadow model: training on representative prompts improves prefetch hit rate.
- Monitor via `/metrics` and tune cache size against observed hit rate.

## Development

    pip install -e .[dev]
    pytest tests/ -v

## Project layout

    mlx_od_moe/
      expert_store.py
      shadow_model.py
      od_moe_layer.py
      model.py
      server.py
      convert/
      training/
    tests/
    examples/
    docs/

## License

MIT. See LICENSE.

## Related

- [MLX](https://github.com/ml-explore/mlx)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF
- [mlx-lm](https://github.com/ml-explore/mlx-examples)
