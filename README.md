# Marathi Sentiment Analysis with Hybrid Deep Learning

Complete sentiment analysis system for Marathi social media text using a hybrid XLM-RoBERTa + CNN architecture.

##  Project Overview

This project implements a comprehensive sentiment analysis system for the **MahaSent dataset** (60,864 Marathi text samples) using multiple approaches:

- **Traditional ML Baselines**: SVM, Random Forest, Logistic Regression, KNN
- **Deep Learning Baselines**: LSTM, BiLSTM, CNN, Multi-CNN
- **PLM Fine-tuning**: XLM-RoBERTa
- **Hybrid Architecture**: XLM-RoBERTa + CNN (main contribution)

### Dataset: MahaSent
- **Total Samples**: 60,864 (perfectly balanced 3-class)
- **Train**: 48,114 samples | **Test**: 6,750 | **Val**: 6,000
- **Labels**: Negative (-1), Neutral (0), Positive (1)
- **Language**: Marathi (Devanagari script)
- **Domain**: Political and social media commentary

##  Architecture

### Hybrid Model Components
1. **Contextual Features**: XLM-RoBERTa base (768-dim embeddings)
2. **Mean Pooling**: Attention-masked mean over sequence
3. **Local Patterns**: Conv1D (768→256, kernel=3)
4. **Feature Fusion**: Concatenate pooled + CNN (1024-dim)
5. **Classification**: Dense layers (1024→512→3) with Dropout(0.3)

##  Project Structure

```
marathi-sentiment/
├── data/                          # Dataset files (CSV)
│   ├── MahaSent_All_Train.csv
│   ├── MahaSent_All_Test.csv
│   └── MahaSent_All_Val.csv
├── src/
│   ├── config.py                  # Configuration (dataclass)
│   ├── data_loader.py             # PyTorch Dataset + DataLoader
│   ├── preprocessor.py            # Marathi text preprocessing
│   ├── train.py                   # Training loop with early stopping
│   ├── evaluate.py                # Metrics and evaluation
│   ├── visualize.py               # Plotting functions
│   ├── models/
│   │   ├── hybrid_model.py        # Main hybrid architecture
│   │   ├── traditional_ml.py      # SVM, RF, LR, KNN
│   │   ├── dl_models.py           # LSTM, BiLSTM, CNN, MCNN
│   │   └── plm_models.py          # PLM fine-tuning wrapper
│   └── utils/
│       └── logger.py              # Experiment tracking
├── notebooks/
│   └── marathi_sentiment_analysis.ipynb  # Main experiment notebook
├── results/                       # Auto-generated outputs
│   ├── figures/                   # Plots and visualizations
│   ├── models/                    # Saved model checkpoints
│   └── logs/                      # Training logs
├── requirements.txt               # Python dependencies
├── DATASET_INSIGHTS_SUMMARY.md    # Dataset analysis
└── README.md                      # This file
```

##  Quick Start

### 1. Installation

```bash
# Clone repository
cd "Maha SA"

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Place your CSV files in the project root:
- `MahaSent_All_Train.csv`
- `MahaSent_All_Test.csv`
- `MahaSent_All_Val.csv`

Expected CSV format:
```csv
text,label
"मराठी text here",-1
"another text",0
"positive text",1
```

### 3. Run the Main Notebook

```bash
jupyter notebook notebooks/marathi_sentiment_analysis.ipynb
```

The notebook includes:
- Data loading and exploration
- Traditional ML baselines
- Deep learning baselines
- PLM fine-tuning
- Hybrid model training
- Comprehensive evaluation
- Visualization
- Error analysis

### 4. Using Individual Modules

```python
from src.config import get_config
from src.data_loader import get_dataloaders_from_config
from src.models.hybrid_model import HybridSentimentModel
from transformers import XLMRobertaTokenizer

# Load configuration
config = get_config()

# Initialize tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(config.plm_name)

# Load data
train_loader, val_loader, test_loader = get_dataloaders_from_config(config, tokenizer)

# Initialize model
model = HybridSentimentModel(config).to(config.device)

# Train model (see src/train.py for full training loop)
```

##  Hyperparameters

### Hybrid Model
- **PLM**: `xlm-roberta-base`
- **Max Sequence Length**: 256 tokens
- **Batch Size**: 64
- **Learning Rate**: 1e-4 (hybrid), 2e-5 (PLM fine-tuning)
- **Epochs**: 20 (with early stopping, patience=3)
- **CNN**: Out channels=256, Kernel=3
- **Dense**: Hidden size=512
- **Dropout**: 0.3

### Traditional ML
- **TF-IDF**: Max features=5000
- **SVM**: Linear kernel, C=1.0, balanced weights
- **Random Forest**: 200 estimators, min_samples_split=6
- **Logistic Regression**: Max iter=1000, balanced weights
- **KNN**: k=5

##  Expected Results

- **Hybrid Model**: >80% accuracy on test set
- **Improvement**: 3-5% F1-score gain over baselines
- **Inference Time**: <50ms per sample
- **Model Size**: ~500MB (XLM-RoBERTa base)

##  Configuration

All hyperparameters are managed through `src/config.py`:

```python
from src.config import Config

config = Config()
config.batch_size = 32          # Modify as needed
config.num_epochs = 15
config.learning_rate_hybrid = 5e-5
```

##  Label Mapping

**Important**: The dataset uses labels {-1, 0, 1}, which are automatically converted to {0, 1, 2} for PyTorch:

- Original: -1 (Negative) → PyTorch: 0
- Original: 0 (Neutral) → PyTorch: 1
- Original: 1 (Positive) → PyTorch: 2

This mapping is handled automatically in `data_loader.py`.

##  Evaluation Metrics

The system calculates:
- **Overall**: Accuracy, Macro/Weighted Precision/Recall/F1
- **Per-Class**: Precision, Recall, F1-score for each sentiment
- **Confusion Matrix**: 3×3 matrix
- **Error Analysis**: Misclassification patterns

##  Visualization

The project generates:
1. Training history (loss + accuracy curves)
2. Confusion matrix heatmap
3. Model comparison bar chart
4. Per-class metrics grouped bar chart
5. Text length distribution analysis

All figures are saved to `results/figures/`.

##  Key Features

-  **Complete Pipeline**: From data loading to evaluation
-  **Multiple Baselines**: Traditional ML + Deep Learning
-  **Hybrid Architecture**: Novel XLM-RoBERTa + CNN combination
-  **Early Stopping**: Prevents overfitting
-  **Comprehensive Metrics**: Multiple evaluation perspectives
-  **Experiment Tracking**: Automatic logging
-  **Visualization**: Publication-ready plots
-  **Type Hints**: Full type annotation throughout
-  **Documentation**: Detailed docstrings

##  Dependencies

Main dependencies:
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `scikit-learn>=1.3.0`
- `pandas>=1.5.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `tqdm>=4.65.0`

See `requirements.txt` for complete list.

## Contributing

This is a research project. Feel free to:
- Report issues
- Suggest improvements
- Extend the architecture
- Add new baselines

##  License

This project is for educational and research purposes.

##  Authors

Sentiment Analysis Research Team

##  Acknowledgments

- **XLM-RoBERTa**: Facebook AI Research
- **MahaSent Dataset**: Original dataset creators
- **Transformers Library**: Hugging Face

##  Contact

For questions or collaboration:
- Open an issue in the repository
- Contact the project maintainers

---

**Note**: Ensure you have sufficient GPU memory (at least 8GB) for training the hybrid model. CPU training is supported but significantly slower.

##  Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in `config.py`
   - Use gradient accumulation
   - Use a smaller model variant

2. **Slow Training**
   - Enable GPU: Check `torch.cuda.is_available()`
   - Reduce `max_seq_length`
   - Use smaller batch sizes

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python version (3.10+)
   - Verify virtual environment activation

```

---
