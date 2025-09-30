# Hanzo Engine Embeddings Implementation

## Overview
Fixed the `/v1/embeddings` endpoint in the Hanzo Engine to generate real embeddings instead of placeholder vectors.

## Changes Made

### 1. Created New Embeddings Module (`/Users/z/work/hanzo/engine/hanzo-engine/src/embeddings.rs`)
- Implemented `EmbeddingEngine` struct that uses the mistralrs BERT model
- Uses the Snowflake Arctic Embed L model for high-quality embeddings
- Supports batch processing of multiple texts
- Implements mean pooling and L2 normalization for sentence embeddings
- Handles various input formats (single string, array of strings, token arrays)

### 2. Updated Main Server (`/Users/z/work/hanzo/engine/hanzo-engine/src/main.rs`)
- Added embeddings module import
- Integrated `EmbeddingEngine` into server state
- Replaced placeholder embedding generation with real implementation
- Added proper error handling and request validation
- Support for optional dimension reduction via `dimensions` parameter

### 3. Updated Cargo Configuration (`/Users/z/work/hanzo/engine/hanzo-engine/Cargo.toml`)
- Changed default features from `cuda` to `metal` for macOS compatibility
- Kept mistralrs dependencies for embedding model support

## Key Features

### Real Embedding Generation
- Uses BERT-based Snowflake Arctic Embed L model
- Generates high-quality semantic embeddings
- Proper tokenization and attention masking
- Mean pooling over token embeddings for sentence representation
- L2 normalization for consistent embedding magnitudes

### API Compatibility
- Fully compatible with OpenAI embeddings API format
- Supports all input types:
  - Single string: `{"input": "text"}`
  - String array: `{"input": ["text1", "text2"]}`
  - Token arrays (converted to strings)
- Optional dimension reduction via `dimensions` parameter
- Proper usage tracking and response formatting

### Error Handling
- Graceful fallback if model fails to load
- Detailed error messages for invalid requests
- Proper HTTP status codes

## Testing

Created test script at `/Users/z/work/hanzo/engine/test_embeddings.py` that:
- Tests single string input
- Tests batch string array input
- Tests dimension reduction
- Validates response structure
- Detects placeholder vs real embeddings

## Usage

### Starting the Server
```bash
cd /Users/z/work/hanzo/engine
cargo run --package hanzo-engine --no-default-features --features metal -- serve
```

### Making Requests
```bash
# Single text
curl -X POST http://localhost:36900/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "model": "snowflake-arctic-embed-l"}'

# Multiple texts
curl -X POST http://localhost:36900/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello", "World"], "model": "snowflake-arctic-embed-l"}'

# With dimension reduction
curl -X POST http://localhost:36900/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Test text", "dimensions": 512}'
```

### Running Tests
```bash
python3 /Users/z/work/hanzo/engine/test_embeddings.py
```

## Technical Details

### Model: Snowflake Arctic Embed L
- State-of-the-art embedding model
- 1024-dimensional embeddings by default
- Excellent performance on semantic similarity tasks
- Supports multiple languages

### Implementation Details
- Mean pooling: Averages token embeddings weighted by attention mask
- L2 normalization: Ensures unit-length embeddings for cosine similarity
- Batch processing: Efficient handling of multiple texts
- Memory efficient: Processes texts one at a time to avoid OOM

## Build Notes

On macOS, use Metal backend instead of CUDA:
```bash
cargo build --package hanzo-engine --no-default-features --features metal
```

For Linux with CUDA:
```bash
cargo build --package hanzo-engine --features cuda
```

## Future Improvements

1. Add support for more embedding models (e.g., different sizes)
2. Implement caching for frequently requested embeddings
3. Add streaming support for large batches
4. Optimize memory usage for very large texts
5. Add support for custom tokenization parameters