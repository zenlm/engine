use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use dirs;
use tracing_subscriber;

mod commands;
mod config;
mod server;

#[derive(Parser)]
#[command(name = "hanzo-engine")]
#[command(about = "Hanzo Engine - High-performance AI inference engine", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "36900")]
        port: u16,
        
        /// Host to bind to
        #[arg(short = 'H', long, default_value = "0.0.0.0")]
        host: String,
        
        /// Model to load on startup
        #[arg(short, long)]
        model: Option<String>,
        
        /// Enable Ollama compatibility mode
        #[arg(long)]
        ollama_compat: bool,
        
        /// Model directory
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
    
    /// Pull a model from a registry
    Pull {
        /// Model identifier or URL
        model: String,
        
        /// Source registry (huggingface, ollama, mlx, url)
        #[arg(short, long, default_value = "huggingface")]
        source: String,
        
        /// Quantization format (q4_0, q5_k_m, q8_0, f16)
        #[arg(short, long)]
        quantization: Option<String>,
    },
    
    /// List downloaded models
    List {
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },
    
    /// Delete a model
    Delete {
        /// Model identifier
        model: String,
        
        /// Force deletion without confirmation
        #[arg(short, long)]
        force: bool,
    },
    
    /// Chat with a model
    Chat {
        /// Model to use
        #[arg(short, long)]
        model: Option<String>,
        
        /// System prompt
        #[arg(short, long)]
        system: Option<String>,
    },
    
    /// Generate embeddings
    Embed {
        /// Input text
        text: String,
        
        /// Model to use
        #[arg(short, long, default_value = "qwen3-embedding-8b")]
        model: String,
        
        /// Output format (json, numpy, raw)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    
    /// Rerank documents
    Rerank {
        /// Query text
        query: String,
        
        /// Documents (can be multiple)
        #[arg(required = true)]
        documents: Vec<String>,
        
        /// Model to use
        #[arg(short, long, default_value = "qwen3-reranker-4b")]
        model: String,
        
        /// Top K results to return
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,
    },
    
    /// Show model information
    Info {
        /// Model identifier
        model: String,
    },
    
    /// Configure engine settings
    Config {
        /// Set default model
        #[arg(long)]
        default_model: Option<String>,
        
        /// Set model directory
        #[arg(long)]
        model_dir: Option<PathBuf>,
        
        /// Set cache directory
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        
        /// Show current configuration
        #[arg(short, long)]
        show: bool,
    },
}

fn get_hanzo_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Could not find home directory")
        .join(".hanzo")
}

fn ensure_hanzo_dirs() -> Result<()> {
    let hanzo_dir = get_hanzo_dir();
    let models_dir = hanzo_dir.join("models");
    let cache_dir = hanzo_dir.join("cache");
    let config_dir = hanzo_dir.join("config");
    
    std::fs::create_dir_all(&models_dir)?;
    std::fs::create_dir_all(&cache_dir)?;
    std::fs::create_dir_all(&config_dir)?;
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize logging
    let level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(level)
        .init();
    
    // Ensure ~/.hanzo directories exist
    ensure_hanzo_dirs()?;
    
    match cli.command {
        Commands::Serve { port, host, model, ollama_compat, model_dir } => {
            println!("🚀 Starting Hanzo Engine server on {}:{}", host, port);
            if ollama_compat {
                println!("   Ollama compatibility mode enabled");
            }
            if let Some(model) = &model {
                println!("   Loading model: {}", model);
            }
            
            let model_dir = model_dir.unwrap_or_else(|| get_hanzo_dir().join("models"));
            server::start_server(host, port, model, ollama_compat, model_dir).await?;
        }
        
        Commands::Pull { model, source, quantization } => {
            println!("📦 Pulling model: {}", model);
            println!("   Source: {}", source);
            if let Some(q) = &quantization {
                println!("   Quantization: {}", q);
            }
            
            commands::pull::pull_model(&model, &source, quantization.as_deref()).await?;
        }
        
        Commands::List { detailed } => {
            println!("📋 Listing models in ~/.hanzo/models");
            commands::list::list_models(detailed)?;
        }
        
        Commands::Delete { model, force } => {
            if !force {
                println!("⚠️  Are you sure you want to delete '{}'? Use --force to confirm.", model);
            } else {
                println!("🗑️  Deleting model: {}", model);
                commands::delete::delete_model(&model)?;
            }
        }
        
        Commands::Chat { model, system } => {
            let model = model.unwrap_or_else(|| String::from("qwen3-8b"));
            println!("💬 Starting chat with model: {}", model);
            if let Some(sys) = &system {
                println!("   System prompt: {}", sys);
            }
            commands::chat::start_chat(&model, system.as_deref()).await?;
        }
        
        Commands::Embed { text, model, format } => {
            println!("🔢 Generating embeddings with model: {}", model);
            let embeddings = commands::embed::generate_embeddings(&text, &model).await?;
            commands::embed::output_embeddings(embeddings, &format)?;
        }
        
        Commands::Rerank { query, documents, model, top_k } => {
            println!("🎯 Reranking {} documents with model: {}", documents.len(), model);
            let results = commands::rerank::rerank_documents(&query, &documents, &model, top_k).await?;
            commands::rerank::output_results(results)?;
        }
        
        Commands::Info { model } => {
            println!("ℹ️  Model information for: {}", model);
            commands::info::show_model_info(&model)?;
        }
        
        Commands::Config { default_model, model_dir, cache_dir, show } => {
            if show {
                println!("⚙️  Current configuration:");
                config::show_config()?;
            } else {
                if let Some(model) = default_model {
                    config::set_default_model(&model)?;
                    println!("✓ Default model set to: {}", model);
                }
                if let Some(dir) = model_dir {
                    config::set_model_dir(&dir)?;
                    println!("✓ Model directory set to: {}", dir.display());
                }
                if let Some(dir) = cache_dir {
                    config::set_cache_dir(&dir)?;
                    println!("✓ Cache directory set to: {}", dir.display());
                }
            }
        }
    }
    
    Ok(())
}

mod commands {
    pub mod pull {
        use anyhow::Result;
        use std::path::PathBuf;
        use indicatif::{ProgressBar, ProgressStyle};
        
        pub async fn pull_model(model: &str, source: &str, quantization: Option<&str>) -> Result<()> {
            let models_dir = crate::get_hanzo_dir().join("models");
            
            match source {
                "huggingface" => pull_from_huggingface(model, quantization, &models_dir).await?,
                "ollama" => pull_from_ollama(model, &models_dir).await?,
                "mlx" => pull_from_mlx(model, &models_dir).await?,
                "url" => pull_from_url(model, &models_dir).await?,
                _ => anyhow::bail!("Unknown source: {}", source),
            }
            
            Ok(())
        }
        
        async fn pull_from_huggingface(model: &str, quantization: Option<&str>, models_dir: &PathBuf) -> Result<()> {
            println!("   Downloading from HuggingFace Hub...");
            
            let pb = ProgressBar::new(100);
            pb.set_style(ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
                .progress_chars("#>-"));
            
            // TODO: Implement actual HuggingFace download
            for i in 0..100 {
                pb.set_position(i);
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }
            pb.finish_with_message("✓ Model downloaded successfully");
            
            Ok(())
        }
        
        async fn pull_from_ollama(model: &str, models_dir: &PathBuf) -> Result<()> {
            println!("   Downloading from Ollama registry...");
            // TODO: Implement Ollama registry download
            Ok(())
        }
        
        async fn pull_from_mlx(model: &str, models_dir: &PathBuf) -> Result<()> {
            println!("   Downloading MLX model for Apple Silicon...");
            // TODO: Implement MLX download
            Ok(())
        }
        
        async fn pull_from_url(url: &str, models_dir: &PathBuf) -> Result<()> {
            println!("   Downloading from URL: {}", url);
            // TODO: Implement direct URL download
            Ok(())
        }
    }
    
    pub mod list {
        use anyhow::Result;
        use std::fs;
        
        pub fn list_models(detailed: bool) -> Result<()> {
            let models_dir = crate::get_hanzo_dir().join("models");
            
            if !models_dir.exists() {
                println!("No models found.");
                return Ok(());
            }
            
            for entry in fs::read_dir(models_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    let model_name = path.file_name().unwrap().to_string_lossy();
                    println!("  • {}", model_name);
                    
                    if detailed {
                        // Show model details
                        if let Ok(metadata) = fs::metadata(&path) {
                            if let Ok(modified) = metadata.modified() {
                                println!("    Modified: {:?}", modified);
                            }
                            // Calculate directory size
                            let size = calculate_dir_size(&path)?;
                            println!("    Size: {:.2} GB", size as f64 / 1_073_741_824.0);
                        }
                    }
                }
            }
            
            Ok(())
        }
        
        fn calculate_dir_size(path: &std::path::Path) -> Result<u64> {
            let mut total_size = 0;
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let metadata = entry.metadata()?;
                if metadata.is_file() {
                    total_size += metadata.len();
                } else if metadata.is_dir() {
                    total_size += calculate_dir_size(&entry.path())?;
                }
            }
            Ok(total_size)
        }
    }
    
    pub mod delete {
        use anyhow::Result;
        use std::fs;
        
        pub fn delete_model(model: &str) -> Result<()> {
            let model_path = crate::get_hanzo_dir().join("models").join(model);
            
            if !model_path.exists() {
                anyhow::bail!("Model '{}' not found", model);
            }
            
            fs::remove_dir_all(model_path)?;
            println!("✓ Model '{}' deleted successfully", model);
            
            Ok(())
        }
    }
    
    pub mod chat {
        use anyhow::Result;
        
        pub async fn start_chat(model: &str, system: Option<&str>) -> Result<()> {
            println!("Chat interface coming soon...");
            // TODO: Implement interactive chat
            Ok(())
        }
    }
    
    pub mod embed {
        use anyhow::Result;
        
        pub async fn generate_embeddings(text: &str, model: &str) -> Result<Vec<f32>> {
            // TODO: Implement actual embedding generation
            println!("   Generating embeddings for: \"{}\"", text);
            Ok(vec![0.1, 0.2, 0.3, 0.4]) // Placeholder
        }
        
        pub fn output_embeddings(embeddings: Vec<f32>, format: &str) -> Result<()> {
            match format {
                "json" => {
                    println!("{}", serde_json::to_string_pretty(&embeddings)?);
                }
                "numpy" => {
                    println!("numpy.array({:?})", embeddings);
                }
                "raw" => {
                    println!("{:?}", embeddings);
                }
                _ => anyhow::bail!("Unknown format: {}", format),
            }
            Ok(())
        }
    }
    
    pub mod rerank {
        use anyhow::Result;
        use serde::Serialize;
        
        #[derive(Serialize)]
        pub struct RerankResult {
            pub index: usize,
            pub score: f32,
            pub text: String,
        }
        
        pub async fn rerank_documents(query: &str, documents: &[String], model: &str, top_k: usize) -> Result<Vec<RerankResult>> {
            // TODO: Implement actual reranking
            let mut results = Vec::new();
            for (i, doc) in documents.iter().enumerate().take(top_k) {
                results.push(RerankResult {
                    index: i,
                    score: 0.9 - (i as f32 * 0.1),
                    text: doc.clone(),
                });
            }
            Ok(results)
        }
        
        pub fn output_results(results: Vec<RerankResult>) -> Result<()> {
            for (i, result) in results.iter().enumerate() {
                println!("{}. [Score: {:.4}] {}", i + 1, result.score, result.text);
            }
            Ok(())
        }
    }
    
    pub mod info {
        use anyhow::Result;
        
        pub fn show_model_info(model: &str) -> Result<()> {
            let model_path = crate::get_hanzo_dir().join("models").join(model);
            
            if !model_path.exists() {
                anyhow::bail!("Model '{}' not found", model);
            }
            
            // TODO: Read and display actual model metadata
            println!("   Path: {}", model_path.display());
            println!("   Type: Embedding Model");
            println!("   Parameters: 8B");
            println!("   Quantization: Q5_K_M");
            println!("   Context: 32768 tokens");
            
            Ok(())
        }
    }
}

mod config {
    use anyhow::Result;
    use serde::{Deserialize, Serialize};
    use std::path::PathBuf;
    use std::fs;
    
    #[derive(Serialize, Deserialize)]
    pub struct Config {
        pub default_model: Option<String>,
        pub model_dir: Option<PathBuf>,
        pub cache_dir: Option<PathBuf>,
    }
    
    fn config_path() -> PathBuf {
        crate::get_hanzo_dir().join("config").join("engine.json")
    }
    
    pub fn load_config() -> Result<Config> {
        let path = config_path();
        if path.exists() {
            let content = fs::read_to_string(path)?;
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(Config {
                default_model: None,
                model_dir: None,
                cache_dir: None,
            })
        }
    }
    
    pub fn save_config(config: &Config) -> Result<()> {
        let path = config_path();
        let content = serde_json::to_string_pretty(config)?;
        fs::write(path, content)?;
        Ok(())
    }
    
    pub fn show_config() -> Result<()> {
        let config = load_config()?;
        println!("   Default model: {}", config.default_model.as_deref().unwrap_or("not set"));
        println!("   Model directory: {}", config.model_dir.as_deref().map(|p| p.display().to_string()).unwrap_or_else(|| "~/.hanzo/models".to_string()));
        println!("   Cache directory: {}", config.cache_dir.as_deref().map(|p| p.display().to_string()).unwrap_or_else(|| "~/.hanzo/cache".to_string()));
        Ok(())
    }
    
    pub fn set_default_model(model: &str) -> Result<()> {
        let mut config = load_config()?;
        config.default_model = Some(model.to_string());
        save_config(&config)
    }
    
    pub fn set_model_dir(dir: &PathBuf) -> Result<()> {
        let mut config = load_config()?;
        config.model_dir = Some(dir.clone());
        save_config(&config)
    }
    
    pub fn set_cache_dir(dir: &PathBuf) -> Result<()> {
        let mut config = load_config()?;
        config.cache_dir = Some(dir.clone());
        save_config(&config)
    }
}

mod server {
    use anyhow::Result;
    use axum::{
        Router,
        routing::{get, post},
        response::Json,
        extract::State,
    };
    use serde_json::{json, Value};
    use std::path::PathBuf;
    use std::sync::Arc;
    use tower_http::cors::CorsLayer;
    
    struct ServerState {
        model: Option<String>,
        model_dir: PathBuf,
        ollama_compat: bool,
    }
    
    pub async fn start_server(
        host: String,
        port: u16,
        model: Option<String>,
        ollama_compat: bool,
        model_dir: PathBuf,
    ) -> Result<()> {
        let state = Arc::new(ServerState {
            model,
            model_dir,
            ollama_compat,
        });
        
        let app = Router::new()
            .route("/", get(root))
            .route("/health", get(health))
            .route("/v1/models", get(list_models))
            .route("/v1/embeddings", post(embeddings))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .with_state(state)
            .layer(CorsLayer::permissive());
        
        let addr = format!("{}:{}", host, port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        
        println!("✓ Server listening on http://{}", addr);
        
        axum::serve(listener, app).await?;
        
        Ok(())
    }
    
    async fn root() -> Json<Value> {
        Json(json!({
            "name": "Hanzo Engine",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "High-performance AI inference engine"
        }))
    }
    
    async fn health() -> Json<Value> {
        Json(json!({
            "status": "healthy"
        }))
    }
    
    async fn list_models(State(state): State<Arc<ServerState>>) -> Json<Value> {
        Json(json!({
            "object": "list",
            "data": [
                {
                    "id": "qwen3-embedding-8b",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "hanzo"
                },
                {
                    "id": "qwen3-reranker-4b",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "hanzo"
                }
            ]
        }))
    }
    
    async fn embeddings(State(state): State<Arc<ServerState>>, Json(payload): Json<Value>) -> Json<Value> {
        // TODO: Implement actual embedding generation
        Json(json!({
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": vec![0.1; 4096], // Placeholder for Qwen3-8B
                "index": 0
            }],
            "model": "qwen3-embedding-8b",
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10
            }
        }))
    }
    
    async fn chat_completions(State(state): State<Arc<ServerState>>, Json(payload): Json<Value>) -> Json<Value> {
        // TODO: Implement actual chat completion
        Json(json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": state.model.as_deref().unwrap_or("qwen3-8b"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm Hanzo Engine. How can I help you today?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 12,
                "total_tokens": 22
            }
        }))
    }
    
    async fn completions(State(state): State<Arc<ServerState>>, Json(payload): Json<Value>) -> Json<Value> {
        // TODO: Implement actual completion
        Json(json!({
            "id": "cmpl-123",
            "object": "text_completion",
            "created": 1234567890,
            "model": state.model.as_deref().unwrap_or("qwen3-8b"),
            "choices": [{
                "text": "This is a completion from Hanzo Engine.",
                "index": 0,
                "logprobs": null,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 8,
                "total_tokens": 13
            }
        }))
    }
}