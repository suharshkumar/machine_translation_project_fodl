# English-to-Hindi Machine Translation using Transformer Architecture

## Overview

This project implements a neural machine translation system for English-to-Hindi translation using a Transformer-based sequence-to-sequence architecture. The system was developed as part of CS6910: Fundamentals of Deep Learning Programming Assignment III, demonstrating advanced deep learning techniques for natural language processing tasks.

## Project Features

### Architecture Highlights
- **Seq2Seq Transformer** with multi-head attention mechanism
- **Multi-GPU Support** using PyTorch DataParallel
- **Mixed Precision Training** with automatic mixed precision (AMP)
- **Gradient Accumulation** for effective large batch training
- **Resumable Training** with comprehensive checkpointing
- **Advanced Evaluation** with multiple BLEU score metrics

### Language Processing
- **Source Language**: English with GloVe 100D word embeddings
- **Target Language**: Hindi with IndicBERT embeddings
- **Vocabulary Management**: Custom vocabulary building with special tokens
- **Tokenization**: Language-specific tokenizers for optimal performance

## Model Architecture

### Transformer Components
- **Encoder Layers**: 2 layers with 4 attention heads
- **Decoder Layers**: 2 layers with 4 attention heads  
- **Hidden Dimension**: 128
- **Feed Forward Dimension**: 512
- **Dropout Rate**: 0.1
- **Positional Encoding**: Sinusoidal positional embeddings

### Embedding Strategy
- **English**: Pre-trained GloVe embeddings (100D, frozen)
- **Hindi**: IndicBERT embeddings (128D, frozen)
- **Projection Layers**: Linear transformations to match model dimensions

## Training Configuration

### Hyperparameters
- **Optimizer**: Adam with learning rate 0.0005
- **Batch Size**: 32 (with gradient accumulation steps: 4)
- **Maximum Sequence Length**: 64 tokens
- **Training Epochs**: 20
- **Gradient Clipping**: 1.0
- **Loss Function**: CrossEntropyLoss with padding token masking

### Training Features
- **Automatic Mixed Precision**: CUDA-enabled gradient scaling
- **Multi-GPU Training**: DataParallel for distributed computation
- **Checkpointing**: Save best models based on validation loss and BLEU scores
- **Progress Tracking**: Comprehensive training history logging

## Evaluation Metrics

### Performance Measures
- **Loss**: CrossEntropy loss with perplexity calculation
- **BLEU Scores**: BLEU@1, BLEU@2, BLEU@3, BLEU@4 with smoothing
- **Translation Quality**: Greedy decoding for inference

### Results Summary
Based on the training results:
- **Final Test Loss**: 3.5802 (PPL: 35.88)
- **BLEU@1**: 0.2691
- **BLEU@2**: 0.1541  
- **BLEU@3**: 0.0964
- **BLEU@4**: 0.0588

## Installation and Setup

### Dependencies
```bash
pip install torch transformers numpy pandas tqdm nltk
```

### Required Downloads
- GloVe embeddings: `glove.6B.100d.txt`
- IndicBERT model: `ai4bharat/indic-bert`
- Training data: English-Hindi parallel corpus

### Data Format
The training data should be in CSV format with columns:
- `source`: English sentences
- `target`: Hindi sentences

## Usage

### 1. Data Preparation
Place your CSV files with the following names:
- `train.csv`: Training data
- `valid.csv`: Validation data
- `test.csv`: Test data

### 2. Training
```bash
python machine_translation.py
```

### 3. Key Files Generated
- `nmt_dp_best_loss.pt`: Best model based on validation loss
- `nmt_dp_training_history.json`: Complete training metrics
- `nmt_exp_FINAL_REPORT_DP_v5.json`: Test set evaluation results

## Project Structure
```
machine-translation/
├── machine_translation.py          # Main implementation
├── requirements.txt                # Dependencies
├── README.md                      # This file
├── data/                          # Dataset directory
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── embeddings/                    # Pre-trained embeddings
│   └── glove.6B.100d.txt
└── checkpoints/                   # Training outputs
    ├── nmt_dp_best_loss.pt
    ├── nmt_dp_training_history.json
    └── nmt_exp_FINAL_REPORT_DP_v5.json
```

## Technical Contributions

1. **Multi-GPU Training**: Efficient DataParallel implementation
2. **Memory Optimization**: Gradient accumulation and mixed precision training
3. **Robust Evaluation**: Multiple BLEU score calculations with smoothing
4. **Language-Specific Processing**: Optimized tokenization for English and Hindi
5. **Comprehensive Logging**: Detailed training history and checkpoint management

## Future Enhancements

- Implement beam search decoding for improved translation quality
- Add attention visualization capabilities
- Experiment with different transformer architectures (T5, mT5)
- Incorporate back-translation for data augmentation
- Add support for other Indian languages using multilingual models

## License

This project is part of academic coursework for CS6910: Fundamentals of Deep Learning.

## Acknowledgments

- Course: CS6910 Fundamentals of Deep Learning
- Institution: IITM
- Semester: January-May 2025

This implementation demonstrates advanced neural machine translation techniques with practical considerations for production deployment, including multi-GPU training, comprehensive evaluation metrics, and robust checkpoint management.
