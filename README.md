# Encoder–Decoder Models: With vs Without Attention Mechanism

> **Assignment:** Review, Implementation, and Comparative Analysis
> **Task:** Abstractive Text Summarization using LSTM Encoder–Decoder on Amazon Fine Food Reviews
> **Reference Paper:** Seq2Seq with Attention for Text Summarization (IJRAR, 2024–2025)

---

## 📌 Overview

This project implements and compares two Sequence-to-Sequence (Seq2Seq) models for abstractive text summarization:

- **Model A** — Standard LSTM Encoder–Decoder (No Attention)
- **Model B** — LSTM Encoder–Decoder with Bahdanau (Additive) Attention

The goal is to demonstrate how the attention mechanism overcomes the information bottleneck of vanilla Seq2Seq models, resulting in better quality summaries and higher BLEU scores.

---

## 📁 Project Structure

```
├── encoder_decoder_001.ipynb   # Main notebook (all 12 cells)
├── comparison_plots.png        # Loss, accuracy & BLEU comparison chart
├── attention_heatmap.png       # Bahdanau attention weight heatmap
└── README.md
```

---

## 🗃️ Dataset

| Detail | Value |
|---|---|
| Source | [Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) |
| Raw Rows Loaded | 50,000 |
| After Filtering & Sampling | 10,000 |
| Input (Encoder) | Review Text — 8 to 80 words |
| Target (Decoder) | Review Summary — 2 to 10 words |
| Train / Test Split | 80% / 20% |

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| Embedding Dimension | 128 |
| LSTM Hidden Units (Latent Dim) | 256 |
| Max Input Length | 80 tokens |
| Max Summary Length | 12 tokens |
| Vocabulary Size | 8,000 |
| Batch Size | 128 |
| Max Epochs | 50 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Cross-Entropy |
| Early Stopping Patience | 5 epochs on val_loss |
| Framework | TensorFlow 2.x / Keras (Google Colab) |

---

## 🏗️ Model Architectures

### Model A — Without Attention

The entire input sequence is compressed into a **single fixed-size vector** (final LSTM hidden state). The decoder only sees this one vector — information bottleneck problem.

```
Input → Embedding → LSTM Encoder → (h, c) final state only
                          ↓
        Decoder LSTM (init with h, c) → Dense(softmax) → Output
```

### Model B — With Bahdanau Attention

The encoder returns **all hidden states**. At each decoder step, the attention layer computes a dynamic context vector by attending over all encoder states.

```
Input → Embedding → LSTM Encoder → ALL hidden states h₁...hₙ
                          ↓
    Bahdanau Attention: score(sₜ, hᵢ) → αᵢ → context_t
                          ↓
    [context_t ; dec_emb] → Decoder LSTM → Dense(softmax) → Output
```

**Attention Score Formula (Bahdanau / Additive):**

```
score(sₜ₋₁, hᵢ) = Vᵀ · tanh(W₁·hᵢ + W₂·sₜ₋₁)
αᵢ  = softmax(score_i)       ← attention weights
context_t = Σ αᵢ · hᵢ        ← dynamic weighted context vector
```

---

## 🚀 How to Run

### Step 1 — Open Notebook in Google Colab

Upload or open `encoder_decoder_001.ipynb` in [Google Colab](https://colab.research.google.com/).

### Step 2 — Add Your Kaggle Credentials

In **Cell 2**, replace the credentials with your own Kaggle API key:

```python
kaggle_creds = {
    'username': 'your_kaggle_username',
    'key':      'your_kaggle_api_key'
}
```

> Get your API key from: https://www.kaggle.com/settings → API → Create New Token

### Step 3 — Run All Cells in Order

| Cell | Description |
|---|---|
| Cell 1 | Install libraries and import dependencies |
| Cell 2 | Download Amazon Fine Food Reviews via Kaggle API |
| Cell 3 | Load, clean, tokenize, pad and split dataset |
| Cell 4 | Build Model A — Encoder-Decoder without Attention |
| Cell 5 | Train Model A with EarlyStopping & ReduceLROnPlateau |
| Cell 6 | Build Model B — Encoder-Decoder with Bahdanau Attention |
| Cell 7 | Train Model B with EarlyStopping & ReduceLROnPlateau |
| Cell 8 | Compute BLEU scores on 200 test samples |
| Cell 9 | Print side-by-side predictions on 8 test samples |
| Cell 10 | Plot loss curves, accuracy curves & comparison bar chart |
| Cell 11 | Generate Bahdanau attention weight heatmap |
| Cell 12 | Print final comparison table of all metrics |

---

## 📊 Results

| Metric | Without Attention | With Attention |
|---|---|---|
| Final Training Loss | — | — |
| Best Validation Loss | — | — |
| Best Validation Accuracy | — | — |
| BLEU Score | — | — |
| Training Time (seconds) | — | — |
| Total Epochs Run | — | — |
| Context Vector | Fixed (bottleneck) | Dynamic (per step) |
| Long Sequence Handling | Poor | Good |
| Interpretability | None | High (heatmap) |
| Output Coherence | Lower | Higher |

> ℹ️ Fill in the numeric values from the Cell 12 output after running the notebook.

---

## 📦 Requirements

```
tensorflow
numpy
pandas
matplotlib
nltk
kaggle
rouge-score
```

Install all at once:

```bash
pip install tensorflow nltk rouge-score kaggle
```

> All other dependencies (numpy, pandas, matplotlib) come pre-installed in Google Colab.

---

## 📄 References

- **Paper:** Seq2Seq with Attention for Text Summarization — IJRAR, 2024–2025
  https://ijrar.org/papers/IJRAR24D2346.pdf
- **Bahdanau et al. (2015):** Neural Machine Translation by Jointly Learning to Align and Translate
  https://arxiv.org/abs/1409.0473
- **Reference Repository:** Harvard NLP Seq2Seq Attention
  https://github.com/harvardnlp/seq2seq-attn
