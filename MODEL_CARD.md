---
license: apache-2.0
tags:
- inference
- llm
- rust
- gguf
- mlx
- quantization
- zen-ai
library_name: mistralrs
---

# Zen Engine - High-Performance Inference

**Zen Engine** is a blazingly fast, Rust-native inference engine powering the entire Zen AI model family. Built on mistral.rs, it provides production-ready inference with OpenAI-compatible APIs.

## Overview

Zen Engine is the unified inference platform for all Zen models:

- **zen-nano** (0.6B) - Ultra-lightweight edge inference
- **zen-eco** (4B) - Efficient instruct/thinking/agent inference
- **zen-agent** (4B) - Tool-calling with MCP support
- **zen-director** (5B) - Text-to-video generation
- **zen-musician** (7B) - Music generation from lyrics
- **zen-3d** (3.3B) - Controllable 3D asset generation

## Key Features

### Performance
- 🚀 **Native Rust**: Maximum performance, low latency
- ⚡ **44K tokens/sec**: M3 Max with MLX backend
- 💨 **28K tokens/sec**: RTX 4090 with GGUF Q4
- 🔥 **ISQ**: In-Situ Quantization for instant model loading
- 📄 **PagedAttention**: Memory-efficient attention mechanism
- ⚡ **FlashAttention**: 2x faster attention computation

### Model Support
- 🔮 **All Zen Models**: Native support for entire family
- 📦 **Multiple Formats**: PyTorch, GGUF, GGML, MLX
- 🎯 **Quantization**: Q2_K to F16, ISQ on-the-fly
- 🌐 **Multimodal**: Text, vision, audio, music, video, 3D

### APIs
- 🔗 **OpenAI Compatible**: Drop-in replacement for OpenAI API
- 🦀 **Rust API**: High-performance native interface
- 🐍 **Python Bindings**: PyO3-based Python support
- 🔌 **MCP Support**: Connect to external tools and services
- 🌐 **HTTP Server**: RESTful API on port 3690

### Backends
- 🔥 **CUDA**: NVIDIA GPU acceleration
- 🍎 **Metal/MLX**: Apple Silicon optimization
- 💻 **CPU**: Fallback for any hardware
- 🔀 **Multi-GPU**: Automatic device mapping

## Supported Quantization Formats

### GGUF (llama.cpp)
- **Q2_K**: 2-bit (smallest, lowest quality)
- **Q3_K_S/M/L**: 3-bit variants
- **Q4_K_S/M**: 4-bit (recommended balance)
- **Q5_K_S/M**: 5-bit (high quality)
- **Q6_K**: 6-bit (near-lossless)
- **Q8_0**: 8-bit (lossless)
- **F16**: 16-bit floating point
- **F32**: 32-bit floating point

### ISQ (In-Situ Quantization)
- **Q2_K, Q3_K, Q4_K**: On-the-fly quantization
- **Q5_K, Q6_K, Q8_0**: Memory-efficient loading
- **HQQ**: Half-Quadratic Quantization

### MLX (Apple Silicon)
- **4-bit, 8-bit**: Optimized for Metal
- **Mixed precision**: Automatic optimization

### GPTQ/AWQ
- **2-bit, 3-bit, 4-bit**: Activation-aware quantization
- **Group size**: 32, 64, 128

## Hardware Requirements

### Minimum
- **CPU**: 4 cores, 8GB RAM
- **GPU**: Optional (CPU inference supported)
- **Storage**: 2GB for zen-nano GGUF Q4

### Recommended
- **GPU**: 16GB VRAM (RTX 4080, RTX 3090)
- **CPU**: 8 cores, 16GB RAM
- **Storage**: 50GB for full model collection

### Optimal
- **GPU**: 24GB+ VRAM (RTX 4090, A100)
- **CPU**: 16+ cores, 32GB+ RAM
- **Storage**: 500GB NVMe SSD

### Apple Silicon
- **M1/M2/M3**: All variants supported
- **M3 Max**: 44K tokens/sec with MLX
- **RAM**: 16GB+ unified memory

## Performance Benchmarks

### Throughput (tokens/sec)

| Model | Hardware | Backend | Quant | Speed |
|-------|----------|---------|-------|-------|
| zen-nano | M3 Max | MLX | 4-bit | 44,000 |
| zen-eco | RTX 4090 | CUDA | Q4_K_M | 28,000 |
| zen-eco | RTX 3090 | CUDA | Q4_K_M | 18,000 |
| zen-agent | M3 Pro | MLX | 4-bit | 32,000 |
| zen-musician | RTX 4090 | CUDA | Q8_0 | 12,000 |

### Latency (time to first token)

| Model | Hardware | Latency |
|-------|----------|---------|
| zen-nano | M3 Max | 8ms |
| zen-eco | RTX 4090 | 12ms |
| zen-agent | RTX 3090 | 18ms |

### Memory Usage

| Model | Format | VRAM/RAM |
|-------|--------|----------|
| zen-nano | Q4_K_M | 0.4GB |
| zen-nano | F16 | 1.2GB |
| zen-eco | Q4_K_M | 2.5GB |
| zen-eco | F16 | 8.0GB |
| zen-agent | Q4_K_M | 2.5GB |
| zen-musician | Q4_K_M | 4.5GB |
| zen-director | Q4_K_M | 3.2GB |

## Installation

### From Source (Rust)

```bash
git clone https://github.com/zenlm/zen-engine.git
cd zen-engine

# Basic build
cargo build --release

# With CUDA
cargo build --release --features "cuda flash-attn"

# With Metal (macOS)
cargo build --release --features metal

# Install binary
cargo install --path .
```

### From Cargo

```bash
cargo install zen-engine --features cuda
```

## Quick Start

### Command Line

```bash
# Start zen-nano
zen-engine serve --model zenlm/zen-nano-0.6b --port 3690

# Start zen-eco with FlashAttention
zen-engine serve --model zenlm/zen-eco-4b-instruct --flash-attn

# Start zen-agent with MCP
zen-engine serve --model zenlm/zen-agent-4b --enable-mcp

# Use GGUF quantized model
zen-engine serve --model zenlm/zen-eco-4b-Q4_K_M.gguf

# Use MLX on Apple Silicon
zen-engine serve --model zenlm/zen-nano-0.6b --backend mlx
```

### OpenAI API (Chat Completions)

```bash
curl -X POST http://localhost:3690/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zen-eco-4b-instruct",
    "messages": [
      {"role": "user", "content": "What is quantum computing?"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3690/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="zen-eco-4b-instruct",
    messages=[
        {"role": "user", "content": "Explain machine learning"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Rust

```rust
use mistralrs::{
    IsqType, PagedAttentionMetaBuilder, TextMessageRole,
    TextModelBuilder, TextMessages
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = TextModelBuilder::new("zenlm/zen-eco-4b-instruct")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(TextMessageRole::User, "What is Rust?");

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content);

    Ok(())
}
```

## Use Cases

### Primary Use Cases
- Production inference for Zen models
- High-throughput API serving
- Edge deployment (zen-nano)
- Real-time applications
- Research and experimentation

### Example Applications
- Chatbot backends
- Code generation services
- Tool-calling agents
- Music generation APIs
- Video generation pipelines
- 3D asset creation services

## Advanced Features

### Multi-GPU Support

```bash
zen-engine serve \
    --model zenlm/zen-eco-4b-instruct \
    --gpu-devices 0,1 \
    --device-map auto
```

### CPU Offloading

```bash
zen-engine serve \
    --model zenlm/zen-musician-7b \
    --cpu-offload \
    --offload-layers 20
```

### Custom Chat Templates

```bash
zen-engine serve \
    --model zenlm/zen-eco-4b-instruct \
    --chat-template ./custom_template.jinja
```

### MCP Tool Integration

```bash
zen-engine serve \
    --model zenlm/zen-agent-4b \
    --enable-mcp \
    --mcp-config ./mcp_config.json
```

## Quantization Guide

### Choosing Quantization

- **Q2_K**: Smallest size, lowest quality (testing only)
- **Q3_K_M**: Small size, acceptable quality
- **Q4_K_M**: ⭐ Best balance (recommended)
- **Q5_K_M**: High quality, larger size
- **Q8_0**: Near-lossless, 2x size of Q4
- **F16**: Full precision, largest size

### Creating GGUF Models

Train with Zen Gym, then quantize:

```bash
# Convert PyTorch to GGUF
python convert.py /path/to/zen-model --outfile zen-model-f16.gguf

# Quantize to Q4_K_M
./quantize zen-model-f16.gguf zen-model-Q4_K_M.gguf Q4_K_M
```

## Limitations

- Some features require specific hardware (e.g., CUDA, Metal)
- Quantization reduces model quality slightly
- Very large models may require multiple GPUs
- MCP support is experimental
- Some multimodal features are in development

## Ethical Considerations

- Inference speed may enable harmful applications at scale
- Users should implement rate limiting and content filtering
- Models may produce biased or harmful outputs
- Proper monitoring and logging recommended
- Consider environmental impact of inference workloads

## Citation

```bibtex
@misc{zenengine2025,
  title={Zen Engine: High-Performance Inference for Zen Models},
  author={Zen AI Team},
  year={2025},
  howpublished={\url{https://github.com/zenlm/zen-engine}}
}
```

## Links

- **GitHub**: https://github.com/zenlm/zen-engine
- **Organization**: https://github.com/zenlm
- **HuggingFace Models**: https://huggingface.co/zenlm
- **Zen Gym** (Training): https://github.com/zenlm/zen-gym
- **Documentation**: See README.md

## Support

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/zenlm/zen-engine/issues
- **Documentation**: See project README

## Acknowledgements

Built on [mistral.rs](https://github.com/EricLBuehler/mistral.rs) by Eric Buehler. We thank the mistral.rs team for their excellent work.

Part of the **[Zen AI](https://github.com/zenlm)** ecosystem - unified AI model training and inference.