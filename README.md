<a name="top"></a>
<h1 align="center">
  Hanzo Engine
</h1>

<h3 align="center">
Native AI inference engine for Hanzo AI - Blazingly fast LLM & embedding inference.
</h3>

<p align="center">
| <a href="https://github.com/hanzoai/engine"><b>GitHub</b></a> | <a href="https://hanzo.ai"><b>Hanzo AI</b></a> | <a href="https://docs.hanzo.ai/engine"><b>Documentation</b></a> | <a href="https://discord.gg/hanzoai"><b>Discord</b></a> |
</p>

<p align="center">
  <a href="https://github.com/hanzoai/engine/stargazers">
    <img src="https://img.shields.io/github/stars/hanzoai/engine?style=social&label=Star" alt="GitHub stars">
  </a>
</p>

**Hanzo Engine is a high-performance, Rust-native inference engine powering Hanzo AI infrastructure:**

- 🚀 **Blazingly Fast**: Native Rust implementation for maximum performance
- 🔮 **All-in-one multimodal**: text↔text, text+vision↔text, text+vision+audio↔text, text→speech, text→image
- 🎯 **Embeddings First-Class**: Optimized for Qwen3-Embedding models (#1 on MTEB)
- 🌐 **APIs**: Rust, Python, OpenAI HTTP server (Chat Completions, Embeddings API)
- 🔗 **MCP Support**: Connect to external tools and services automatically
- ⚡ **Performance**: ISQ, PagedAttention, FlashAttention, MLX support for Apple Silicon
- 🔧 **Default Port**: 36900 (Hanzo Node connects on port 3690)

## Quick Start 🚀

### Installation

```bash
# Clone the repository
git clone https://github.com/hanzoai/engine.git
cd engine

# Build the engine
cargo build --release

# Or install via cargo
cargo install --git https://github.com/hanzoai/engine
```

### Running the Engine

```bash
# Start the engine server (default port: 36900)
hanzo-engine serve

# With custom port
hanzo-engine serve --port 36900

# With specific model
hanzo-engine serve --model qwen3-embedding-8b
```

### Embeddings API

```bash
# Generate embeddings
curl -X POST http://localhost:36900/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-8b",
    "input": "Hello, Hanzo Engine!"
  }'
```

### Supported Models

#### Embedding Models (Optimized)
- **Qwen3-Embedding-8B**: 4096 dims, #1 on MTEB multilingual
- **Qwen3-Embedding-4B**: 2048 dims, balanced performance
- **Qwen3-Embedding-0.6B**: 1024 dims, lightweight

#### Reranker Models
- **Qwen3-Reranker-4B**: Superior reranking quality
- **Qwen3-Reranker-0.6B**: Lightweight reranking

#### LLM Models
- All models supported by original mistral.rs
- Llama, Mistral, Phi, Gemma, and more

## Integration with Hanzo Node

Hanzo Engine is the core inference backend for [Hanzo Node](https://github.com/hanzoai/node):

```rust
// In your Hanzo Node configuration
export EMBEDDINGS_SERVER_URL="http://localhost:36900"  // Engine port
export EMBEDDING_MODEL_TYPE="qwen3-embedding-8b"
export USE_NATIVE_EMBEDDINGS="true"
```

## Model Management

### Pull Models

```bash
# Pull from HuggingFace
hanzo-engine pull qwen3-embedding-8b --source huggingface

# Pull GGUF format
hanzo-engine pull https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/main/qwen3-8b-q5_k_m.gguf

# Pull MLX format (Apple Silicon)
hanzo-engine pull qwen3-embedding-8b --source mlx

# List downloaded models
hanzo-engine list

# Delete a model
hanzo-engine delete qwen3-embedding-8b
```

## Performance Features

### Apple Silicon Optimization (MLX)
- 44K+ tokens/second on M-series chips
- Optimized matrix operations
- Reduced memory usage

### CUDA Support
- Multi-GPU inference
- Flash Attention v2
- Tensor parallelism

### Quantization
- 4-bit, 5-bit, 8-bit quantization
- Dynamic quantization
- ISQ (In-Situ Quantization)

## API Compatibility

### OpenAI Compatible
- `/v1/chat/completions`
- `/v1/embeddings`
- `/v1/models`

### Ollama Compatible (Port 11434)
For backward compatibility with Ollama clients:
```bash
hanzo-engine serve --ollama-compat --port 11434
```

## Development

### Building from Source

```bash
# Clone with submodules
git clone --recursive https://github.com/hanzoai/engine.git
cd engine

# Build with features
cargo build --release --features cuda,flash-attn,mkl

# Run tests
cargo test
```

### Python Bindings

```python
from hanzo_engine import Engine

# Initialize engine
engine = Engine(model="qwen3-embedding-8b")

# Generate embeddings
embeddings = engine.embed("Hello, Hanzo!")

# Chat completion
response = engine.chat([
    {"role": "user", "content": "What is Hanzo Engine?"}
])
```

## Architecture

```
┌─────────────────────────────────────┐
│         Hanzo Node (3690)           │
│    (AI Agent Infrastructure)        │
└────────────┬────────────────────────┘
             │ HTTP/gRPC
             ▼
┌─────────────────────────────────────┐
│      Hanzo Engine (36900)           │
│   (Native Inference Backend)        │
├─────────────────────────────────────┤
│  • Embedding Generation             │
│  • LLM Inference                    │
│  • Reranking                        │
│  • Model Management                 │
└─────────────────────────────────────┘
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Hanzo Engine is a fork of [mistral.rs](https://github.com/EricLBuehler/mistral.rs) by Eric Buehler, enhanced with enterprise features for Hanzo AI infrastructure.

---

<p align="center">
  Built with ❤️ by <a href="https://hanzo.ai">Hanzo AI</a>
</p>