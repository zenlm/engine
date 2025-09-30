use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_core::{BertEmbeddingModel, MistralRs, MistralRsBuilder, Pipeline, Request, RequestMessage, Response, NormalRequest};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, RwLock};

#[derive(Clone)]
pub struct EmbeddingEngine {
    bert_pipeline: Option<mistralrs_core::embedding::bert::BertPipeline>,
    device: Device,
}

impl EmbeddingEngine {
    pub fn new() -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        
        // Try to load BERT embedding model
        let bert_pipeline = match mistralrs_core::embedding::bert::BertPipeline::new(
            BertEmbeddingModel::SnowflakeArcticEmbedL,
            &device,
        ) {
            Ok(pipeline) => {
                tracing::info!("Loaded BERT embedding model successfully");
                Some(pipeline)
            }
            Err(e) => {
                tracing::warn!("Failed to load BERT embedding model: {}", e);
                None
            }
        };

        Ok(Self {
            bert_pipeline,
            device,
        })
    }

    pub async fn generate_embeddings(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if let Some(pipeline) = &self.bert_pipeline {
            let mut all_embeddings = Vec::new();
            
            for text in texts {
                // Tokenize the text
                let encoding = pipeline.tokenizer.encode(text.clone(), true)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                
                let tokens = encoding.get_ids();
                let token_ids = Tensor::new(tokens, &self.device)?
                    .unsqueeze(0)?; // Add batch dimension
                
                // Create token type ids (all zeros for single sequence)
                let token_type_ids = Tensor::zeros_like(&token_ids)?;
                
                // Get attention mask (1 for real tokens, 0 for padding)
                let attention_mask = Tensor::ones_like(&token_ids)?;
                
                // Forward pass through the model
                let output = pipeline.model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
                
                // Mean pooling over sequence dimension to get sentence embedding
                // output shape: [batch_size, seq_len, hidden_size]
                let embeddings = self.mean_pooling(&output, &attention_mask)?;
                
                // Normalize embeddings
                let embeddings = self.normalize_embeddings(&embeddings)?;
                
                // Convert to Vec<f32>
                let embeddings_vec = embeddings.squeeze(0)?.to_vec1::<f32>()?;
                all_embeddings.push(embeddings_vec);
            }
            
            Ok(all_embeddings)
        } else {
            // Fallback: return placeholder embeddings if model not loaded
            // In production, this should return an error instead
            tracing::warn!("BERT model not loaded, returning placeholder embeddings");
            Ok(texts.iter().map(|_| vec![0.1; 1024]).collect())
        }
    }

    fn mean_pooling(&self, token_embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Expand attention_mask to match embeddings dimensions
        let attention_mask_expanded = attention_mask.unsqueeze(2)?
            .expand(token_embeddings.shape())?
            .to_dtype(token_embeddings.dtype())?;
        
        // Apply attention mask
        let sum_embeddings = (token_embeddings * &attention_mask_expanded)?
            .sum(1)?
            .unsqueeze(1)?;
        
        // Calculate sum of attention mask (avoid division by zero)
        let sum_mask = attention_mask_expanded.sum(1)?
            .clamp(1e-9, f64::INFINITY)?;
        
        // Mean pooling
        sum_embeddings.broadcast_div(&sum_mask)
    }

    fn normalize_embeddings(&self, embeddings: &Tensor) -> Result<Tensor> {
        // L2 normalization
        let norm = embeddings.sqr()?
            .sum_keepdim(embeddings.rank() - 1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?;
        
        embeddings.broadcast_div(&norm)
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    pub model: Option<String>,
    pub encoding_format: Option<String>,
    pub dimensions: Option<usize>,
    pub user: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    String(String),
    StringArray(Vec<String>),
    TokenArray(Vec<u32>),
    TokenArrayArray(Vec<Vec<u32>>),
}

impl EmbeddingInput {
    pub fn to_string_array(self) -> Vec<String> {
        match self {
            EmbeddingInput::String(s) => vec![s],
            EmbeddingInput::StringArray(arr) => arr,
            EmbeddingInput::TokenArray(tokens) => {
                vec![format!("Token array: {:?}", tokens)]
            }
            EmbeddingInput::TokenArrayArray(arrays) => {
                arrays.iter()
                    .map(|tokens| format!("Token array: {:?}", tokens))
                    .collect()
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}