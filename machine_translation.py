"""
English-to-Hindi Machine Translation using Transformer Architecture
CS6910: Fundamentals of Deep Learning - Programming Assignment III

This implementation features:
- Seq2Seq Transformer with multi-head attention
- Multi-GPU training with PyTorch DataParallel
- Mixed precision training with automatic mixed precision (AMP)
- Gradient accumulation for effective batch training
- Comprehensive checkpointing for resumable training
- Advanced evaluation with multiple BLEU score metrics
- GloVe embeddings for English and IndicBERT embeddings for Hindi
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import random
import math
import time
import re
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm.auto import tqdm
import json
from torch.amp import autocast, GradScaler

# ==============================================================================
# CONFIGURATION AND HYPERPARAMETERS
# ==============================================================================

# Set device and reproducibility
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
# Print environment info
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
ngpus_available = torch.cuda.device_count()
print(f"GPUs Available: {ngpus_available}")
print(f"Primary Device: {DEVICE}")

# Special tokens for English
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# Hyperparameters
hyperparameters = {
    'HID_DIM': 128,
    'ENC_LAYERS': 2,
    'DEC_LAYERS': 2,
    'ENC_HEADS': 4,
    'DEC_HEADS': 4,
    'ENC_PF_DIM': 512,
    'DEC_PF_DIM': 512,
    'ENC_DROPOUT': 0.1,
    'DEC_DROPOUT': 0.1,
    'LEARNING_RATE': 0.0005,
    'CLIP': 1.0,
    'BATCH_SIZE': 32,
    'ACCUMULATION_STEPS': 4,
    'MAX_LEN': 64,
    'EPOCHS': 20,
    'DATALOADER_WORKERS': 4, 
    'LOG_INTERVAL': 50,
    'GLOVE_DIM': 100,
    'SEED': SEED
}

# Checkpoint settings
CHECKPOINT_DIR = "./checkpoints/"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
LAST_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "nmt_dp_last_checkpoint.pt")
BEST_LOSS_MODEL_FILE = os.path.join(CHECKPOINT_DIR, "nmt_dp_best_loss.pt")
BEST_BLEU_MODEL_FILE = os.path.join(CHECKPOINT_DIR, "nmt_dp_best_bleu4.pt")
HISTORY_FILE = os.path.join(CHECKPOINT_DIR, "nmt_dp_training_history.json")

# ==============================================================================
# VOCABULARY AND TOKENIZATION
# ==============================================================================

class Vocabulary:
    """English vocabulary management with special tokens"""
    def __init__(self, name, tokenizer_func):
        self.name = name
        self.tokenizer_func = tokenizer_func
        self.token2index = {
            PAD_TOKEN: PAD_IDX,
            SOS_TOKEN: SOS_IDX, 
            EOS_TOKEN: EOS_IDX,
            UNK_TOKEN: UNK_IDX
        }
        self.index2token = {v: k for k, v in self.token2index.items()}
        self.n_tokens = len(self.token2index)
    
    def add_sentence(self, sentence):
        """Add all tokens from a sentence to the vocabulary"""
        tokens = self.tokenizer_func(sentence)
        for token in tokens:
            self.add_token(token)
    
    def add_token(self, token):
        """Add a single token to the vocabulary if not already present"""
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1
    
    def tokens_to_indices(self, tokens):
        """Convert a list of tokens to their indices"""
        return [self.token2index.get(t, UNK_IDX) for t in tokens]
    
    def indices_to_tokens(self, indices, remove_special=True):
        """Convert a list of indices back to tokens, optionally removing special tokens"""
        output_tokens = []
        for idx in indices:
            idx = idx.item() if not isinstance(idx, int) else idx
            if remove_special:
                if idx == PAD_IDX:
                    continue
                if idx == SOS_IDX:
                    continue
                if idx == EOS_IDX:
                    break
            output_tokens.append(self.index2token.get(idx, UNK_TOKEN))
        return output_tokens

def tokenize_english(text):
    """English tokenization with preprocessing"""
    text = str(text).lower()
    text = re.sub(r"([?.!,¿-])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    text = text.strip()
    return text.split(' ')

# ==============================================================================
# DATASET AND DATA LOADING
# ==============================================================================

class TranslationDataset(Dataset):
    """Dataset class for English-Hindi translation pairs"""
    def __init__(self, src_sentences, trg_sentences, src_vocab, indic_tokenizer, 
                 src_tokenizer_func, max_len):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.indic_tokenizer = indic_tokenizer
        self.src_tokenizer_func = src_tokenizer_func
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_text = self.src_sentences[idx]
        trg_text = self.trg_sentences[idx]
        
        # Process English source
        src_tokens = [SOS_TOKEN] + self.src_tokenizer_func(src_text)[:self.max_len-2] + [EOS_TOKEN]
        src_indices = torch.tensor(self.src_vocab.tokens_to_indices(src_tokens), dtype=torch.long)
        
        # Process Hindi target using IndicBERT tokenizer
        hindi_tokens = self.indic_tokenizer.tokenize(trg_text)[:self.max_len-1]
        trg_input_ids = torch.tensor([self.indic_tokenizer.cls_token_id] + 
                                   self.indic_tokenizer.convert_tokens_to_ids(hindi_tokens), 
                                   dtype=torch.long)
        trg_output_ids = torch.tensor(self.indic_tokenizer.convert_tokens_to_ids(hindi_tokens) + 
                                    [self.indic_tokenizer.sep_token_id], 
                                    dtype=torch.long)
        
        return {
            "src_text_orig": src_text,
            "trg_text_orig": trg_text,
            "src_indices": src_indices,
            "trg_input_ids": trg_input_ids,
            "trg_output_ids": trg_output_ids
        }

def collate_fn(batch, eng_pad_idx, hindi_pad_idx):
    """Collate function for batching with padding"""
    src_indices = [item["src_indices"] for item in batch]
    trg_input_ids = [item["trg_input_ids"] for item in batch]
    trg_output_ids = [item["trg_output_ids"] for item in batch]
    src_texts = [item["src_text_orig"] for item in batch]
    trg_texts = [item["trg_text_orig"] for item in batch]
    
    src_padded = pad_sequence(src_indices, batch_first=True, padding_value=eng_pad_idx)
    trg_input_padded = pad_sequence(trg_input_ids, batch_first=True, padding_value=hindi_pad_idx)
    trg_output_padded = pad_sequence(trg_output_ids, batch_first=True, padding_value=hindi_pad_idx)
    
    return {
        "src_text": src_texts,
        "trg_text": trg_texts,
        "src": src_padded,
        "trg_input": trg_input_padded,
        "trg_output": trg_output_padded
    }

def load_data_from_csv(file_path, max_rows=None):
    """Load parallel corpus from CSV file"""
    try:
        df = pd.read_csv(file_path, nrows=max_rows)
        if 'source' not in df.columns or 'target' not in df.columns:
            if 's' in df.columns and 't' in df.columns:
                df = df.rename(columns={'s': 'source', 't': 'target'})
            else:
                raise ValueError(f"CSV must have 'source'/'target' columns")
        
        df.dropna(subset=['source', 'target'], inplace=True)
        src_sentences = df['source'].astype(str).tolist()
        trg_sentences = df['target'].astype(str).tolist()
        
        print(f"Loaded {len(src_sentences)} sentence pairs from {file_path}")
        return src_sentences, trg_sentences
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# ==============================================================================
# EMBEDDING LOADING
# ==============================================================================

def download_glove_if_needed(glove_url="http://nlp.stanford.edu/data/glove.6B.zip", 
                            target_file="glove.6B.100d.txt"):
    """Download GloVe embeddings if not already present"""
    if not os.path.exists(target_file):
        print(f"GloVe file '{target_file}' not found.")
        print(f"Downloading GloVe from {glove_url}...")
        zip_file = "glove_download.zip"
        dl_status = os.system(f"wget -q {glove_url} -O {zip_file}")
        if dl_status != 0:
            print(f"ERROR: Failed to download GloVe zip (status: {dl_status}).")
            return False
        else:
            print("Unzipping GloVe...")
            unzip_cmd = f"unzip -o -q {zip_file} {target_file} -d ."
            uz_status = os.system(unzip_cmd)
            if not os.path.exists(target_file):
                print(f"Specific file extract failed. Trying generic unzip...")
                uz_status = os.system(f"unzip -o -q {zip_file} -d .")
                if not os.path.exists(target_file):
                    print(f"ERROR: Failed to find '{target_file}' after unzipping.")
                    return False
            print(f"'{target_file}' successfully extracted.")
            if os.path.exists(zip_file):
                os.remove(zip_file)  # Clean up zip file
            return True
    else:
        print(f"Using existing GloVe file: {target_file}")
        return True

def load_glove_embeddings(glove_file_path, vocab, embedding_dim):
    """Load GloVe embeddings for English vocabulary"""
    print(f"Loading GloVe embeddings from: {glove_file_path}")
    embeddings_index = {}
    
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading GloVe"):
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except ValueError:
                continue
            if len(coefs) == embedding_dim:
                embeddings_index[word] = coefs
    
    # Initialize embedding matrix
    embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab.n_tokens, embedding_dim)).astype(np.float32)
    embedding_matrix[PAD_IDX] = np.zeros(embedding_dim, dtype=np.float32)
    
    loaded_count = 0
    for token, i in vocab.token2index.items():
        if token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
            continue
        embedding_vector = embeddings_index.get(token)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            loaded_count += 1
    
    print(f"Initialized {loaded_count}/{vocab.n_tokens - 4} words from GloVe")
    return torch.tensor(embedding_matrix, dtype=torch.float)

# ==============================================================================
# TRANSFORMER MODEL ARCHITECTURE
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    """Sequence-to-Sequence Transformer for machine translation"""
    def __init__(self, num_encoder_layers, num_decoder_layers, src_emb_dim, 
                 tgt_emb_dim, d_model, nhead, src_vocab_size, tgt_vocab_size,
                 dim_feedforward, dropout, src_embedding_weights, 
                 tgt_embedding_weights, src_pad_idx, tgt_pad_idx, max_len):
        super().__init__()
        
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, src_emb_dim, padding_idx=src_pad_idx)
        if src_embedding_weights is not None:
            self.src_embedding.weight.data.copy_(src_embedding_weights)
            self.src_embedding.weight.requires_grad = False
        
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, tgt_emb_dim, padding_idx=tgt_pad_idx)
        if tgt_embedding_weights is not None:
            self.tgt_embedding.weight.data.copy_(tgt_embedding_weights)
            self.tgt_embedding.weight.requires_grad = False
        
        # Projection layers to match d_model
        self.src_projection = nn.Linear(src_emb_dim, d_model) if src_emb_dim != d_model else nn.Identity()
        self.tgt_projection = nn.Linear(tgt_emb_dim, d_model) if tgt_emb_dim != d_model else nn.Identity()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
    
    def generate_square_subsequent_mask(self, size, device):
        """Generate causal mask for decoder"""
        mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate_padding_mask(self, seq, pad_idx):
        """Generate padding mask"""
        return (seq == pad_idx)
    
    def forward(self, src, tgt_input):
        device = src.device
        
        # Generate masks
        src_padding_mask = self.generate_padding_mask(src, self.src_pad_idx).to(device)
        tgt_padding_mask = self.generate_padding_mask(tgt_input, self.tgt_pad_idx).to(device)
        
        # Embeddings with scaling and positional encoding
        src_emb = self.src_projection(self.src_embedding(src))
        tgt_emb = self.tgt_projection(self.tgt_embedding(tgt_input))
        
        src_emb = self.pos_encoding(src_emb * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(tgt_emb * math.sqrt(self.d_model))
        
        # Causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.shape[1], device)
        
        # Transformer forward pass
        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask
        )
        
        return self.output_projection(output)

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ==============================================================================

def train_epoch(model, dataloader, optimizer, criterion, clip_value, device, 
                current_epoch, log_interval, accumulation_steps=1, scaler=None):
    """Train model for one epoch with gradient accumulation"""
    model.train()
    epoch_loss = 0.0
    num_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {current_epoch} Training")
    
    for batch_idx, batch in enumerate(pbar):
        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        
        src = batch["src"].to(device)
        trg_input = batch["trg_input"].to(device)
        trg_output = batch["trg_output"].to(device)
        
        # Forward pass with mixed precision
        with autocast(device_type=device.type, enabled=(scaler is not None)):
            output_logits = model(src, trg_input)
            loss = criterion(output_logits.view(-1, output_logits.shape[-1]), 
                           trg_output.view(-1))
            
            if isinstance(model, nn.DataParallel):
                loss = loss.mean()
            
            loss = loss / accumulation_steps
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        actual_loss = loss.item() * accumulation_steps
        epoch_loss += actual_loss * src.size(0)
        num_samples += src.size(0)
        
        # Optimizer step with gradient clipping
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            if scaler:
                scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            current_loss = actual_loss / src.size(0) if src.size(0) > 0 else 0
            pbar.set_postfix_str(f"Loss: {current_loss:.4f}")
    
    return epoch_loss / num_samples if num_samples > 0 else 0.0

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model and return loss"""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            src = batch["src"].to(device)
            trg_input = batch["trg_input"].to(device)
            trg_output = batch["trg_output"].to(device)
            
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                output_logits = model(src, trg_input)
                loss = criterion(output_logits.view(-1, output_logits.shape[-1]),
                               trg_output.view(-1))
                
                if isinstance(model, nn.DataParallel):
                    loss = loss.mean()
            
            total_loss += loss.item() * src.size(0)
            num_samples += src.size(0)
    
    return total_loss / num_samples if num_samples > 0 else 0.0

def translate_sentence_greedy(src_tensor, model, indic_tokenizer, device, max_length):
    """Greedy decoding for translation"""
    model.eval()
    sos_id = indic_tokenizer.cls_token_id
    eos_id = indic_tokenizer.sep_token_id
    
    with torch.no_grad():
        src_tensor = src_tensor.to(device)
        translated_ids = [sos_id]
        
        for _ in range(max_length):
            tgt_input = torch.LongTensor(translated_ids).unsqueeze(0).to(device)
            
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                output_logits = model(src_tensor, tgt_input)
            
            predicted_id = output_logits.argmax(2)[:, -1].item()
            translated_ids.append(predicted_id)
            
            if predicted_id == eos_id:
                break
        
        # Convert to tokens and remove special tokens
        generated_tokens = indic_tokenizer.convert_ids_to_tokens(translated_ids[1:])
        if generated_tokens and generated_tokens[-1] == indic_tokenizer.sep_token:
            generated_tokens = generated_tokens[:-1]
        
        return generated_tokens

def evaluate_with_bleu(model, dataloader, criterion, device, indic_tokenizer, max_length):
    """Evaluate model with BLEU scores"""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    hypotheses = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="BLEU Evaluation"):
            src = batch["src"].to(device)
            trg_input = batch["trg_input"].to(device)
            trg_output = batch["trg_output"].to(device)
            tgt_texts = batch["trg_text"]
            
            # Calculate loss
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                output_logits = model(src, trg_input)
                loss = criterion(output_logits.view(-1, output_logits.shape[-1]),
                               trg_output.view(-1))
                
                if isinstance(model, nn.DataParallel):
                    loss = loss.mean()
            
            total_loss += loss.item() * src.size(0)
            num_samples += src.size(0)
            
            # Generate translations for BLEU calculation
            model_for_translation = model.module if isinstance(model, nn.DataParallel) else model
            
            for i in range(src.size(0)):
                hypothesis_tokens = translate_sentence_greedy(
                    src[i:i+1, :], model_for_translation, indic_tokenizer, device, max_length
                )
                reference_tokens = indic_tokenizer.tokenize(tgt_texts[i])
                
                hypotheses.append(hypothesis_tokens)
                references.append([reference_tokens])
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    # Calculate BLEU scores
    bleu_scores = [0.0, 0.0, 0.0, 0.0]
    if hypotheses and references:
        smoothing = SmoothingFunction().method1
        try:
            bleu_scores[0] = corpus_bleu(references, hypotheses, weights=(1,0,0,0), smoothing_function=smoothing)
            bleu_scores[1] = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0), smoothing_function=smoothing)
            bleu_scores[2] = corpus_bleu(references, hypotheses, weights=(0.33,0.33,0.33,0), smoothing_function=smoothing)
            bleu_scores[3] = corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothing)
        except ZeroDivisionError:
            print("BLEU calculation error - ZeroDivisionError")
    
    return avg_loss, bleu_scores[0], bleu_scores[1], bleu_scores[2], bleu_scores[3]

# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def main_training_loop(train_file, valid_file, test_file, glove_file, 
                      checkpoint_dir="./checkpoints"):
    """Main training function with complete pipeline"""
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load IndicBERT tokenizer
    print("Loading IndicBERT tokenizer...")
    indic_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    hindi_pad_id = indic_tokenizer.pad_token_id if indic_tokenizer.pad_token_id is not None else 0
    
    # Load IndicBERT embeddings
    print("Loading IndicBERT model for embeddings...")
    indic_model = AutoModel.from_pretrained("ai4bharat/indic-bert")
    indic_embedding_weights = indic_model.get_input_embeddings().weight.data.clone()
    indic_vocab_size = indic_embedding_weights.size(0)
    indic_embedding_dim = indic_embedding_weights.size(1)
    
    # Load training data and build English vocabulary
    print("Loading training data...")
    train_src, train_trg = load_data_from_csv(train_file)
    
    print("Building English vocabulary...")
    eng_vocab = Vocabulary("english", tokenize_english)
    for sentence in tqdm(train_src, desc="Building vocabulary"):
        eng_vocab.add_sentence(sentence)
    
    print(f"English vocabulary size: {eng_vocab.n_tokens}")
    
    # Load GloVe embeddings
    english_embeddings = load_glove_embeddings(glove_file, eng_vocab, hyperparameters['GLOVE_DIM'])
    
    # Load all datasets
    valid_src, valid_trg = load_data_from_csv(valid_file)
    test_src, test_trg = load_data_from_csv(test_file)
    
    # Create datasets and dataloaders
    train_dataset = TranslationDataset(train_src, train_trg, eng_vocab, indic_tokenizer, 
                                     tokenize_english, hyperparameters['MAX_LEN'])
    valid_dataset = TranslationDataset(valid_src, valid_trg, eng_vocab, indic_tokenizer,
                                     tokenize_english, hyperparameters['MAX_LEN'])
    test_dataset = TranslationDataset(test_src, test_trg, eng_vocab, indic_tokenizer,
                                    tokenize_english, hyperparameters['MAX_LEN'])
    
    # Collate function with proper padding indices
    def custom_collate_fn(batch):
        return collate_fn(batch, PAD_IDX, hindi_pad_id)
    
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparameters['BATCH_SIZE'],
                                shuffle=True, collate_fn=custom_collate_fn, 
                                num_workers=hyperparameters['DATALOADER_WORKERS'])
    valid_dataloader = DataLoader(valid_dataset, batch_size=hyperparameters['BATCH_SIZE'],
                                shuffle=False, collate_fn=custom_collate_fn, 
                                num_workers=hyperparameters['DATALOADER_WORKERS'])
    test_dataloader = DataLoader(test_dataset, batch_size=hyperparameters['BATCH_SIZE'],
                               shuffle=False, collate_fn=custom_collate_fn, 
                               num_workers=hyperparameters['DATALOADER_WORKERS'])
    
    # Initialize model
    print("Initializing model...")
    model = Seq2SeqTransformer(
        num_encoder_layers=hyperparameters['ENC_LAYERS'],
        num_decoder_layers=hyperparameters['DEC_LAYERS'],
        src_emb_dim=hyperparameters['GLOVE_DIM'],
        tgt_emb_dim=indic_embedding_dim,
        d_model=hyperparameters['HID_DIM'],
        nhead=hyperparameters['ENC_HEADS'],
        src_vocab_size=eng_vocab.n_tokens,
        tgt_vocab_size=indic_vocab_size,
        dim_feedforward=hyperparameters['ENC_PF_DIM'],
        dropout=hyperparameters['ENC_DROPOUT'],
        src_embedding_weights=english_embeddings.to(DEVICE),
        tgt_embedding_weights=indic_embedding_weights.to(DEVICE),
        src_pad_idx=PAD_IDX,
        tgt_pad_idx=hindi_pad_id,
        max_len=hyperparameters['MAX_LEN'] + 5
    ).to(DEVICE)
    
    # DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Initialize optimizer, criterion, and scaler
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss(ignore_index=hindi_pad_id)
    scaler = GradScaler() if DEVICE.type == 'cuda' else None
    
    print(f"Model parameters: {count_parameters(model.module if isinstance(model, nn.DataParallel) else model):,}")
    
    # Resume from checkpoint if available
    current_epoch = 1
    best_valid_loss = float('inf')
    best_bleu4 = 0.0
    training_history = []
    
    last_checkpoint_path = os.path.join(checkpoint_dir, "nmt_dp_last_checkpoint.pt")
    if os.path.exists(last_checkpoint_path):
        print(f"Resuming from checkpoint: {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path, map_location=DEVICE)
        
        # Load model state
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        current_epoch = checkpoint['epoch'] + 1
        best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
        best_bleu4 = checkpoint.get('best_bleu4', 0.0)
        
        # Load scaler state if available
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training history
        history_path = os.path.join(checkpoint_dir, "nmt_dp_training_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                training_history = json.load(f)
        
        print(f"Resuming from epoch {current_epoch}, best valid loss: {best_valid_loss:.4f}, best BLEU@4: {best_bleu4:.4f}")
    
    # Training loop
    for epoch in range(current_epoch, hyperparameters['EPOCHS'] + 1):
        print(f"\nEpoch {epoch}/{hyperparameters['EPOCHS']}")
        start_time = time.time()
        
        # Training
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion,
                               hyperparameters['CLIP'], DEVICE, epoch, hyperparameters['LOG_INTERVAL'],
                               hyperparameters['ACCUMULATION_STEPS'], scaler)
        
        # Validation
        valid_loss = evaluate_model(model, valid_dataloader, criterion, DEVICE)
        
        # Calculate perplexity
        train_ppl = math.exp(min(train_loss, 700))
        valid_ppl = math.exp(min(valid_loss, 700))
        
        # Evaluate with BLEU scores every few epochs (or on final epoch)
        if epoch % 5 == 0 or epoch == hyperparameters['EPOCHS']:
            print("Evaluating with BLEU scores...")
            valid_loss, bleu1, bleu2, bleu3, bleu4 = evaluate_with_bleu(
                model, valid_dataloader, criterion, DEVICE, indic_tokenizer, hyperparameters['MAX_LEN']
            )
            print(f"BLEU Scores: B@1={bleu1:.4f} | B@2={bleu2:.4f} | B@3={bleu3:.4f} | B@4={bleu4:.4f}")
        else:
            # Default values if not calculating BLEU every epoch
            bleu1, bleu2, bleu3, bleu4 = 0.0, 0.0, 0.0, 0.0
        
        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) - (epoch_mins * 60))
        
        print(f"Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"Valid Loss: {valid_loss:.4f} | Valid PPL: {valid_ppl:.2f}")
        
        # Save epoch results
        epoch_stats = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'valid_loss': round(valid_loss, 4),
            'train_ppl': round(train_ppl, 2),
            'valid_ppl': round(valid_ppl, 2),
            'bleu1': round(bleu1, 4),
            'bleu2': round(bleu2, 4),
            'bleu3': round(bleu3, 4),
            'bleu4': round(bleu4, 4),
            'time_mins': epoch_mins,
            'time_secs': epoch_secs
        }
        
        # Update training history
        # Remove existing entry for this epoch if exists (for resuming training)
        training_history = [e for e in training_history if e.get('epoch') != epoch]
        training_history.append(epoch_stats)
        training_history.sort(key=lambda x: x['epoch'])
        
        # Save best model based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_model_path = os.path.join(checkpoint_dir, "nmt_dp_best_loss.pt")
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(base_model.state_dict(), best_loss_model_path)
            print(f"New best validation loss! Model saved to {best_loss_model_path}")
        
        # Save best model based on BLEU score (if calculated)
        if bleu4 > best_bleu4 and bleu4 > 0:
            best_bleu4 = bleu4
            best_bleu_model_path = os.path.join(checkpoint_dir, "nmt_dp_best_bleu4.pt")
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(base_model.state_dict(), best_bleu_model_path)
            print(f"New best BLEU@4 score! Model saved to {best_bleu_model_path}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_valid_loss': best_valid_loss,
            'best_bleu4': best_bleu4
        }
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        checkpoint_path = os.path.join(checkpoint_dir, "nmt_dp_last_checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save training history
        history_path = os.path.join(checkpoint_dir, "nmt_dp_training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Clear GPU cache
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Final evaluation on test set with BLEU scores
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load best model
    best_model_path = os.path.join(checkpoint_dir, "nmt_dp_best_loss.pt")
    if os.path.exists(best_model_path):
        final_model = Seq2SeqTransformer(
            num_encoder_layers=hyperparameters['ENC_LAYERS'],
            num_decoder_layers=hyperparameters['DEC_LAYERS'],
            src_emb_dim=hyperparameters['GLOVE_DIM'],
            tgt_emb_dim=indic_embedding_dim,
            d_model=hyperparameters['HID_DIM'],
            nhead=hyperparameters['ENC_HEADS'],
            src_vocab_size=eng_vocab.n_tokens,
            tgt_vocab_size=indic_vocab_size,
            dim_feedforward=hyperparameters['ENC_PF_DIM'],
            dropout=hyperparameters['ENC_DROPOUT'],
            src_embedding_weights=english_embeddings.to(DEVICE),
            tgt_embedding_weights=indic_embedding_weights.to(DEVICE),
            src_pad_idx=PAD_IDX,
            tgt_pad_idx=hindi_pad_id,
            max_len=hyperparameters['MAX_LEN'] + 5
        ).to(DEVICE)
        
        final_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        
        # Evaluate with BLEU scores
        test_loss, bleu1, bleu2, bleu3, bleu4 = evaluate_with_bleu(
            final_model, test_dataloader, criterion, DEVICE, indic_tokenizer, hyperparameters['MAX_LEN']
        )
        
        test_ppl = math.exp(min(test_loss, 700))
        
        print(f"Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.2f}")
        print(f"BLEU Scores:")
        print(f"  BLEU@1: {bleu1:.4f}")
        print(f"  BLEU@2: {bleu2:.4f}")
        print(f"  BLEU@3: {bleu3:.4f}")
        print(f"  BLEU@4: {bleu4:.4f}")
        
        # Generate example translations
        print("\n" + "="*60)
        print("EXAMPLE TRANSLATIONS")
        print("="*60)
        
        final_model.eval()
        for i in range(min(5, len(test_dataset))):
            sample = test_dataset[i]
            src_text = sample["src_text_orig"]
            tgt_text = sample["trg_text_orig"]
            src_tensor = sample["src_indices"].unsqueeze(0).to(DEVICE)
            
            translated_tokens = translate_sentence_greedy(
                src_tensor, final_model, indic_tokenizer, DEVICE, hyperparameters['MAX_LEN']
            )
            translated_text = indic_tokenizer.decode(
                indic_tokenizer.convert_tokens_to_ids(translated_tokens), 
                skip_special_tokens=True
            )
            
            print(f"\nExample {i+1}:")
            print(f"Source (EN): {src_text}")
            print(f"Target (HI): {tgt_text}")
            print(f"Model Output (HI): {translated_text}")
            print("-" * 40)
        
        # Save final results
        final_results = {
            "test_results": {
                "loss": test_loss,
                "perplexity": test_ppl,
                "bleu1": bleu1,
                "bleu2": bleu2,
                "bleu3": bleu3,
                "bleu4": bleu4
            },
            "hyperparameters": hyperparameters,
            "training_history": training_history
        }
        
        results_path = os.path.join(checkpoint_dir, "nmt_exp_FINAL_REPORT.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\nTraining completed! Final results saved to {results_path}")
        return final_results
    else:
        print(f"Best model not found at {best_model_path}")
        return None

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Example usage - replace with your actual file paths
    train_file = "data/train.csv"
    valid_file = "data/valid.csv" 
    test_file = "data/test.csv"
    glove_file = "embeddings/glove.6B.100d.txt"
    
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Download GloVe if needed
    if download_glove_if_needed(target_file=glove_file):
        # Run training
        results = main_training_loop(train_file, valid_file, test_file, glove_file)
        print("Training completed successfully!")
    else:
        print("Failed to setup GloVe embeddings. Cannot proceed with training.")