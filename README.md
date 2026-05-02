🏦 Sentiment & Aspect Analysis of Bank Reviews
Using IndoBERT + Bi-LSTM | Multi-Label Text Classification

📌 Overview

This project is my undergraduate thesis — a deep dive into understanding **what Indonesian bank customers actually feel** when they write reviews. Not just "positive" or "negative", but *which specific aspect* they're talking about: service quality, application performance, interest rates, customer support, and more.

The system combines the contextual power of **IndoBERT** (a BERT model pre-trained on Indonesian text) with the sequential learning capability of **Bi-LSTM** to classify bank reviews across multiple service aspects simultaneously.

> **TL;DR**: Give it a customer review in Indonesian → it tells you the sentiment AND which service aspects are being discussed.

🎯 Problem Statement

Standard sentiment analysis gives you a single label: *positive*, *negative*, or *neutral*. But a real bank review is rarely that simple:

> "Aplikasinya mudah dipakai tapi bunga pinjamannya terlalu tinggi dan CS susah dihubungi."

This sentence has three aspects — app usability (positive), loan interest (negative), and customer service (negative). A single label misses all of that nuance.

This project tackles that with **multi-label aspect-based sentiment analysis**.

🧠 Model Architecture

```
Input Text (Indonesian)
        ↓
[ IndoBERT Tokenizer ]
        ↓
[ IndoBERT Encoder ] ← Pre-trained on Indonesian corpus
        ↓
  Contextualized Embeddings
        ↓
[ Bidirectional LSTM ]
        ↓
  Sequential Features (forward + backward)
        ↓
[ Fully Connected + Sigmoid ]
        ↓
  Multi-Label Output (per aspect)
```

The hybrid architecture leverages:
- IndoBERT → captures deep semantic and contextual meaning in Indonesian language
- Bi-LSTM → models sequential dependencies in both directions
- Sigmoid activationn → enables independent multi-label prediction per aspect

📂 Project Structure

```
📦 Sentiment-Aspect-Analysis-Bank-Reviews
├── 📄 .gitignore
├── 📄 Fusion.py                                  # Model fusion logic (IndoBERT + Bi-LSTM)
├── 📄 Merge.py                                   # Dataset merging utility
├── 📄 Scrap2.py                                  # Google Maps review scraper
├── 📄 TA_MultiModel_ADASYN_Final.ipynb           # Main training notebook (with ADASYN oversampling)
├── 📄 analisis-ulasan-bank.py                    # Bank review analysis script
├── 📄 sampling.py                                # Data sampling strategy
│
├── 📊 aspect_fold_metrics.csv                    # Per-fold metrics for aspect classification
├── 📊 aspect_metrics_plot.png                    # Visualization of aspect model performance
├── 📊 rating_fold_metrics.csv                    # Per-fold metrics for rating/sentiment
├── 📊 rating_metrics_plot.png                    # Visualization of rating model performance
├── 📊 predictions_aspek_raw.csv                  # Raw model predictions on aspect labels
│
├── 📋 bank_lateng2.csv                           # Raw scraped dataset (batch 2)
├── 📋 bank_lateng3.csv                           # Raw scraped dataset (batch 3)
├── 📋 bank_lateng_merged.csv                     # Merged & cleaned dataset
├── 📋 bank_mandiri.csv                           # Bank Mandiri specific reviews
├── 📋 dataset_Google-Maps-Reviews-Scraper_2025-0...  # Full Google Maps dataset
│
└── 📄 README.md
```

⚙️ Pipeline
1. Data Collection (`Scrap2.py`)
- Scraping reviews from Google Maps using automated scraper
- Collected from multiple Central Java (`bank_lateng`) & national banks (e.g. Bank Mandiri)
- Raw data stored in CSV format per batch, then merged via `Merge.py`

2. Data Preprocessing & Sampling (`sampling.py`, `analisis-ulasan-bank.py`)
- Text normalization (lowercasing, punctuation removal)
- Indonesian slang/abbreviation handling
- Tokenization using IndoBERT tokenizer
- Class imbalance handling with **ADASYN** (Adaptive Synthetic Sampling)

3. Model Training (`TA_MultiModel_ADASYN_Final.ipynb`, `Fusion.py`)
- Feature extraction via `indobenchmark/indobert-base-p1`
- Hybrid architecture: IndoBERT embeddings → Bi-LSTM layers
- Optimizer: AdamW with linear warmup scheduler
- Loss: Binary Cross-Entropy with Logits (multi-label)
- K-Fold cross-validation for robust evaluation

4. Evaluation & Results
- Per-fold metrics saved to `aspect_fold_metrics.csv` & `rating_fold_metrics.csv`
- Performance visualized in `aspect_metrics_plot.png` & `rating_metrics_plot.png`
- Raw predictions exported to `predictions_aspek_raw.csv`

🚀 Getting Started
### Prerequisites
```bash
Python >= 3.8
pip install -r requirements.txt
```

### Install Dependencies
```bash
pip install torch transformers scikit-learn pandas numpy seaborn
```

### Run Training
```bash
python src/train.py --epochs 10 --batch_size 16 --lr 2e-5
```

### Run Evaluation
```bash
python src/evaluate.py --model_path models/best_model.pt
```

🔧 Key Challenges & Solutions

| Challenge | Solution |
|---|---|
| Class imbalance across aspect labels | Weighted Binary Cross-Entropy loss |
| Limited labeled Indonesian banking data | Data augmentation + careful manual labeling |
| Overfitting on small dataset | Dropout, early stopping, cross-validation |
| Hyperparameter sensitivity | Grid search + validation loss monitoring |

📚 References
- Koto et al. (2020) — *IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP*
- Devlin et al. (2019) — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- Pontiki et al. (2016) — *SemEval-2016 Task 5: Aspect Based Sentiment Analysis*

👤 Author
Wahyu Ariyadi  
Computer Science — Universitas Sebelas Maret  
Bangkit Academy 2023 Graduate (Batch 2, GOTO)


📄 License
This project is licensed under the MIT License — feel free to use it for learning and research purposes.
