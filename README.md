<a name="top"></a>
<h1 align="center">
  Zen Engine
</h1>

<h3 align="center">
Native AI inference engine for Zen models - Blazingly fast LLM, embedding, and multimodal inference.
</h3>

<p align="center">
| <a href="https://github.com/zenlm/zen-engine"><b>GitHub</b></a> | <a href="https://zenlm.ai"><b>Zen AI</b></a> | <a href="https://docs.zenlm.ai/engine"><b>Documentation</b></a> |
</p>

<p align="center">
  <a href="https://github.com/zenlm/zen-engine/stargazers">
    <img src="https://img.shields.io/github/stars/zenlm/zen-engine?style=social&label=Star" alt="GitHub stars">
  </a>
</p>

**Zen Engine is a high-performance, Rust-native inference engine powering the entire Zen model family:**

- 🚀 **Blazingly Fast**: Native Rust implementation for maximum performance
- 🔮 **All Zen Models**: Optimized for zen-nano, zen-eco, zen-agent, zen-director, zen-musician
- 🎯 **Multimodal Native**: text↔text, text+vision↔text, text→speech, text→music, text→video
- 🌐 **APIs**: Rust, Python, OpenAI HTTP server (Chat Completions, Embeddings API)
- 🔗 **MCP Support**: Connect to external tools and services automatically
- ⚡ **Performance**: ISQ, PagedAttention, FlashAttention, MLX support for Apple Silicon
- 🔧 **Default Port**: 3690 (Zen model serving)

## Quick Start 🚀

### Installation

```bash
# Clone the repository
git clone https://github.com/zenlm/zen-engine.git
cd zen-engine

# Build the engine
cargo build --release

# Or install via cargo
cargo install --git https://github.com/zenlm/zen-engine
```

### Running Zen Models

```bash
# Start zen-nano (0.6B)
zen-engine serve --model zenlm/zen-nano-0.6b --port 3690

# Start zen-eco instruct (4B)
zen-engine serve --model zenlm/zen-eco-4b-instruct --port 3690

# Start zen-agent with tools (4B)
zen-engine serve --model zenlm/zen-agent-4b --enable-mcp --port 3690

# Start zen-musician (7B) for music generation
zen-engine serve --model zenlm/zen-musician-7b --port 3690

# With MLX on Apple Silicon
zen-engine serve --model zenlm/zen-nano-0.6b --backend mlx

# With GGUF quantization
zen-engine serve --model zenlm/zen-eco-4b-instruct-Q4_K_M.gguf
```

### Chat Completions API

```bash
# Chat with zen-eco
curl -X POST http://localhost:3690/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zen-eco-4b-instruct",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### Embeddings API (Qwen3-Embedding)

```bash
# Generate embeddings with Qwen3
curl -X POST http://localhost:3690/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-8b",
    "input": "Zen models are fast and efficient"
  }'
```

## Supported Zen Models

### Zen Model Family

#### zen-nano (0.6B)
- **Architecture**: Qwen3-0.6B
- **Formats**: PyTorch, MLX, GGUF (Q2_K, Q4_K_M, Q8_0, F16)
- **Use Case**: Edge devices, mobile, low-latency inference
- **Memory**: 1-2GB

#### zen-eco (4B)
- **Variants**: instruct, thinking, agent
- **Architecture**: Qwen3-4B
- **Formats**: PyTorch, MLX, GGUF (Q2_K, Q4_K_M, Q5_K_M, Q8_0, F16)
- **Use Case**: General purpose, balanced performance
- **Memory**: 4-8GB

#### zen-agent (4B)
- **Architecture**: Qwen3-4B + tool-calling
- **Dataset**: Salesforce/xlam-function-calling-60k
- **Formats**: PyTorch, MLX, GGUF
- **Use Case**: Tool use, function calling, API integration
- **Memory**: 4-8GB

#### zen-director (5B)
- **Architecture**: Wan2.2-TI2V-5B
- **Modality**: Text/Image → Video
- **Formats**: PyTorch (Diffusion)
- **Use Case**: Video generation
- **Memory**: 12-16GB

#### zen-musician (7B)
- **Architecture**: YuE-s1-7B
- **Modality**: Lyrics → Music (vocals + accompaniment)
- **Formats**: PyTorch, LoRA adapters
- **Use Case**: Music generation
- **Memory**: 16-24GB

### Embedding Models (Optimized)
- **Qwen3-Embedding-8B**: 4096 dims, #1 on MTEB multilingual
- **Qwen3-Embedding-4B**: 2048 dims, balanced performance
- **Qwen3-Embedding-0.6B**: 1024 dims, lightweight

### Reranker Models
- **Qwen3-Reranker-4B**: Superior reranking quality
- **Qwen3-Reranker-0.6B**: Lightweight reranking

## Model Management

### Pull Models

```bash
# Pull from HuggingFace
zen-engine pull zenlm/zen-nano-0.6b --source huggingface

# Pull GGUF format
zen-engine pull zenlm/zen-eco-4b-instruct-Q4_K_M.gguf

# Pull MLX format (Apple Silicon)
zen-engine pull zenlm/zen-nano-0.6b --source mlx

# List downloaded models
zen-engine list

# Delete a model
zen-engine delete zen-nano-0.6b
```

## Performance Features

### Apple Silicon Optimization (MLX)
- 44K+ tokens/second on M-series chips
- Optimized for zen-nano and zen-eco
- Reduced memory usage
- Native 4.5-bit quantization

### CUDA Support
- Multi-GPU inference
- Flash Attention v2
- Tensor parallelism
- Optimized for RTX 3060+

### Quantization Support
- **GGUF**: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32
- **Dynamic**: Runtime quantization
- **ISQ**: In-Situ Quantization
- **BitDelta**: Efficient delta compression
- **AWQ/GPTQ**: Post-training quantization

## API Compatibility

### OpenAI Compatible
- `/v1/chat/completions` - Chat with zen models
- `/v1/embeddings` - Generate embeddings
- `/v1/models` - List available models
- Drop-in replacement for OpenAI SDK

### Ollama Compatible (Port 11434)
For backward compatibility with Ollama clients:
```bash
zen-engine serve --ollama-compat --port 11434
```

## Integration Examples

### Python SDK

```python
from zen_engine import ZenEngine

# Initialize engine
engine = ZenEngine(
    model="zenlm/zen-eco-4b-instruct",
    device="cuda",
    quantization="Q4_K_M"
)

# Generate text
response = engine.generate(
    "Explain the theory of relativity",
    max_tokens=500,
    temperature=0.7
)
print(response)

# Generate embeddings
embeddings = engine.embed([
    "First sentence",
    "Second sentence"
])
```

### Rust API

```rust
use zen_engine::{ZenEngine, ModelConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load model
    let engine = ZenEngine::new(ModelConfig {
        model: "zenlm/zen-nano-0.6b".to_string(),
        device: "cuda:0".to_string(),
        quantization: Some("Q4_K_M".to_string()),
        ..Default::default()
    }).await?;

    // Generate
    let response = engine.generate(
        "Write a haiku about AI",
        None
    ).await?;

    println!("{}", response);
    Ok(())
}
```

### OpenAI SDK (Drop-in)

```python
from openai import OpenAI

# Point to zen-engine
client = OpenAI(
    base_url="http://localhost:3690/v1",
    api_key="not-needed"
)

# Use any zen model
response = client.chat.completions.create(
    model="zen-eco-4b-instruct",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

## Performance Benchmarks

### Inference Speed (tokens/second)

| Model | Format | Device | Speed |
|-------|--------|--------|-------|
| zen-nano-0.6b | MLX | M3 Max | 44,000 |
| zen-nano-0.6b | GGUF Q4 | M3 Max | 32,000 |
| zen-nano-0.6b | GGUF Q4 | RTX 4090 | 28,000 |
| zen-eco-4b | MLX | M3 Max | 18,000 |
| zen-eco-4b | GGUF Q4 | RTX 4090 | 12,000 |
| zen-eco-4b | GGUF Q4 | RTX 3060 | 5,500 |

### Memory Usage

| Model | Format | VRAM/RAM |
|-------|--------|----------|
| zen-nano-0.6b | F16 | 1.2GB |
| zen-nano-0.6b | Q4_K_M | 0.4GB |
| zen-eco-4b | F16 | 8GB |
| zen-eco-4b | Q4_K_M | 2.3GB |
| zen-eco-4b | Q2_K | 1.6GB |

## Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/zenlm/zen-engine.git
cd zen-engine

# Basic build
cargo build --release

# With CUDA support (Linux)
cargo build --release --features "cuda flash-attn cudnn"

# With Metal support (macOS)
cargo build --release --features metal

# With all features
cargo build --release --features "cuda metal flash-attn"
```

### Running Tests

```bash
# Run all tests
cargo test --workspace

# Run specific package tests
cargo test -p zen-engine-core
cargo test -p zen-engine-quant

# Run benchmarks
cargo bench
```

### Code Quality

```bash
# Format code
cargo fmt --all

# Run clippy
cargo clippy --workspace --tests -- -D warnings

# Check formatting
cargo fmt --all -- --check
```

## Architecture

### Workspace Structure
- `zen-engine-core/` - Core inference engine, Zen model implementations
- `zen-engine-server/` - CLI binary and HTTP server
- `zen-engine-python/` - Python bindings (PyO3)
- `zen-engine-vision/` - Vision model support (zen-director)
- `zen-engine-audio/` - Audio processing (zen-musician)
- `zen-engine-quant/` - Quantization (GGUF, AWQ, GPTQ, BitDelta)
- `zen-engine-paged-attn/` - PagedAttention implementation
- `zen-engine-mcp/` - Model Context Protocol (zen-agent)

### Key Features

1. **Unified Model Loading**: All Zen models load through standardized pipelines
2. **Multi-Backend**: CUDA, Metal, CPU with automatic device selection
3. **Zero-Copy**: Efficient memory sharing between Rust and Python
4. **Async First**: Tokio-based async runtime for concurrent requests
5. **Type Safety**: Strongly typed Rust with Python bindings

## Configuration

### Environment Variables

```bash
# Server configuration
export ZEN_ENGINE_PORT=3690
export ZEN_ENGINE_HOST="0.0.0.0"

# Model paths
export ZEN_MODELS_PATH="/path/to/models"
export ZEN_CACHE_DIR="/path/to/cache"

# Device configuration
export ZEN_DEVICE="cuda:0"
export ZEN_DEVICE_MAP="auto"

# Performance
export ZEN_BATCH_SIZE=32
export ZEN_MAX_CONCURRENT=10
```

### Config File (zen-engine.toml)

```toml
[server]
port = 3690
host = "0.0.0.0"
max_concurrent = 10

[models]
cache_dir = "/path/to/cache"
default_quantization = "Q4_K_M"

[cuda]
devices = [0, 1]
flash_attn = true

[mlx]
enabled = true
```

## Deployment

### Docker

```bash
# Build image
docker build -t zen-engine .

# Run container
docker run -p 3690:3690 \
  -v ./models:/models \
  zen-engine serve --model zenlm/zen-eco-4b-instruct
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zen-engine
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: zen-engine
        image: zenlm/zen-engine:latest
        ports:
        - containerPort: 3690
        env:
        - name: ZEN_DEVICE
          value: "cuda:0"
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Troubleshooting

### Common Issues

**Out of Memory**
```bash
# Use smaller quantization
zen-engine serve --model zen-eco-4b-Q2_K.gguf

# Enable CPU offload
zen-engine serve --model zen-eco-4b --cpu-offload
```

**Slow Inference**
```bash
# Enable FlashAttention
zen-engine serve --model zen-eco-4b --flash-attn

# Use MLX on Apple Silicon
zen-engine serve --model zen-eco-4b --backend mlx
```

**CUDA Errors**
```bash
# Check CUDA version
nvidia-smi

# Rebuild with correct CUDA
cargo clean
cargo build --release --features "cuda flash-attn"
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Credits

Zen Engine is built on [mistral.rs](https://github.com/EricLBuehler/mistral.rs) by Eric Buehler. We thank the mistral.rs team for their excellent work on Rust-native LLM inference.

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@misc{zenengine2025,
  title={Zen Engine: High-Performance Inference for Zen Models},
  author={Zen AI Team},
  year={2025},
  howpublished={\url{https://github.com/zenlm/zen-engine}}
}
```

---

**Zen Engine** - Blazingly fast inference for all Zen models