import os
import re
import logging
import torch
import torch.nn.functional as F
from typing import List
from transformers import AutoTokenizer, AutoModel

class SolutionSimilarityChecker:
    """Handles embedding and similarity computation for solutions"""
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger('similarity_checker')
        
        # Set environment variable to get better CUDA error messages
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Default configuration values if no config is provided
        embedding_model = "sentence-transformers/all-mpnet-base-v2"
        embedding_max_length = 512
        embedding_device = "cpu"
        embedding_fallback_to_cpu = True
        
        # Use config values if provided
        if config:
            embedding_model = getattr(config, 'embedding_model', embedding_model)
            embedding_max_length = getattr(config, 'embedding_max_length', embedding_max_length)
            embedding_device = getattr(config, 'embedding_device', embedding_device)
            embedding_fallback_to_cpu = getattr(config, 'embedding_fallback_to_cpu', embedding_fallback_to_cpu)
        
        # Try to use GPU first, with fallback to CPU if needed
        try:
            # Determine device
            if embedding_device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(embedding_device)
                
            self.logger.info(f"Loading similarity model: {embedding_model} on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                embedding_model,
                use_fast=True,  # Use faster tokenizer implementation
                cache_dir="./.cache/huggingface"  # Cache models locally
            )
            
            # Load model directly to target device with optimizations
            self.model = AutoModel.from_pretrained(
                embedding_model,
                cache_dir="./.cache/huggingface",
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,  # Use half precision on GPU
                low_cpu_mem_usage=True  # Optimize memory usage
            )
            
            # Check if configured max_length exceeds model's capacity
            model_max_length = self.tokenizer.model_max_length
            if embedding_max_length > model_max_length:
                self.logger.warning(
                    f"Configured embedding_max_length ({embedding_max_length}) exceeds model's "
                    f"maximum context length ({model_max_length}). Using model's maximum instead."
                )
                self.max_length = model_max_length
            else:
                self.max_length = embedding_max_length
            
            # Explicitly disable gradient computation
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Move model to device and set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Verify model is on correct device
            if next(self.model.parameters()).device != self.device:
                self.logger.warning(f"Model not on expected device. Moving to {self.device}")
                self.model = self.model.to(self.device)
            
            self.logger.info(f"Similarity model loaded successfully on device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading similarity model on {self.device}: {str(e)}")
            if embedding_fallback_to_cpu and self.device.type == 'cuda':
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
                
                # Try loading on CPU instead
                self.model = AutoModel.from_pretrained(
                    config.embedding_model,
                    cache_dir="./.cache/huggingface",
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
                
                # Disable gradients
                for param in self.model.parameters():
                    param.requires_grad = False
                
                self.model.eval()
                self.logger.info(f"Successfully loaded model on CPU as fallback")
            else:
                raise
        
        # Set batch size - smaller for GPU to avoid OOM
        self.batch_size = getattr(config, 'embedding_batch_size', 8) if config else 8
        if self.device.type == 'cuda':
            self.batch_size = max(1, self.batch_size)
            self.logger.info(f"Using batch size {self.batch_size} for {self.device.type}")

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for a list of texts, processing in batches if needed"""
        if not texts:
            return torch.tensor([], device=self.device)
            
        # Process in batches to avoid OOM errors with larger models
        all_embeddings = []
        
        with torch.no_grad():  # Ensure no gradients are tracked
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                try:
                    # Tokenize with padding and truncation using safe max_length
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    
                    # Move inputs to model device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Use autocast for GPU to improve performance and stability
                    if self.device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(**inputs)
                            token_embeddings = outputs.last_hidden_state
                    else:
                        outputs = self.model(**inputs)
                        token_embeddings = outputs.last_hidden_state
                    
                    # Get attention mask to properly average token embeddings
                    attention_mask = inputs['attention_mask']
                    
                    # Mean pooling with attention mask
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    
                    # Safe operations with explicit detach
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                    
                    # Normalize embeddings
                    normalized = F.normalize(embeddings, p=2, dim=1)
                    
                    # Safety check for NaN/Inf values
                    if torch.isnan(normalized).any() or torch.isinf(normalized).any():
                        self.logger.warning(f"NaN/Inf values detected in batch {i}, replacing with zeros")
                        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    all_embeddings.append(normalized)
                    
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and self.device.type == 'cuda':
                        self.logger.warning(f"CUDA OOM in batch {i}, processing on CPU instead")
                        # Move inputs to CPU and process there
                        cpu_inputs = {k: v.to('cpu') for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            cpu_model = self.model.to('cpu')
                            outputs = cpu_model(**cpu_inputs)
                            token_embeddings = outputs.last_hidden_state
                            
                            # Move model back to original device
                            self.model = self.model.to(self.device)
                            
                            # Continue processing on CPU
                            attention_mask = cpu_inputs['attention_mask']
                            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                            embeddings = sum_embeddings / sum_mask
                            normalized = F.normalize(embeddings, p=2, dim=1)
                            
                            # Move result back to original device
                            normalized = normalized.to(self.device)
                            all_embeddings.append(normalized)
                    else:
                        self.logger.error(f"Error processing batch {i}: {str(e)}")
                        # Return zeros for this batch
                        batch_size = len(batch_texts)
                        embedding_dim = self.model.config.hidden_size
                        all_embeddings.append(torch.zeros(batch_size, embedding_dim, device=self.device))
                
                except Exception as e:
                    self.logger.error(f"Error processing batch {i}: {str(e)}")
                    # Return zeros for this batch
                    batch_size = len(batch_texts)
                    embedding_dim = self.model.config.hidden_size
                    all_embeddings.append(torch.zeros(batch_size, embedding_dim, device=self.device))
            
            # Concatenate all batch embeddings
            if len(all_embeddings) > 1:
                return torch.cat(all_embeddings, dim=0)
            elif len(all_embeddings) == 1:
                return all_embeddings[0]
            else:
                return torch.tensor([], device=self.device)

    def compute_similarity_matrix(self, solutions: List[str]) -> torch.Tensor:
        """Compute pairwise similarities between solutions"""
        if not solutions:
            return torch.tensor([], device=self.device)
            
        with torch.no_grad():
            try:
                # Get embeddings for all solutions
                embeddings = self.get_embeddings(solutions)
                
                # Safety check for NaN or Inf values
                if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                    self.logger.warning("Found NaN or Inf values in embeddings, replacing with zeros")
                    embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # NOTE: Always compute on GPU when possible for performance (CPU is too slow)
                
                # Compute similarity matrix on current device (preferably GPU)
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        similarity_matrix = torch.matmul(embeddings, embeddings.t())
                else:
                    # Fallback to CPU only if necessary
                    cpu_embeddings = embeddings.detach().cpu()
                    similarity_matrix = torch.matmul(cpu_embeddings, cpu_embeddings.t())
                    similarity_matrix = similarity_matrix.to(self.device)
                
                # Ensure values are in valid range [0,1]
                similarity_matrix = torch.clamp(similarity_matrix, 0.0, 1.0)
                
                # Log the matrix shape and a sample
                logging.getLogger('similarity_checker').info(f"Computed similarity matrix with shape: {similarity_matrix.shape}")
                if similarity_matrix.shape[0] > 0:
                    sample_size = min(3, similarity_matrix.shape[0])
                    sample = similarity_matrix[:sample_size, :sample_size]
                    logging.getLogger('similarity_checker').info(f"Sample of similarity matrix:\n{sample}")
                
                return similarity_matrix
                
            except Exception as e:
                self.logger.error(f"Error computing similarity matrix: {str(e)}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Return identity matrix as fallback (each solution only similar to itself)
                return torch.eye(len(solutions), device=self.device)
