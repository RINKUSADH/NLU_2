# Natural Language Understanding - Programming Assignment 2

This repository contains the complete implementation pipeline for learning localized domain-specific Word Embeddings and autoregressive Character-level Sequence Generative Models native to PyTorch.

---

## 📂 Project Structure

### Data Extraction \& Embeddings (Problem 1)
* **`fetch_corpus.py`**: A custom web-crawler pipeline that strictly targets official IIT Jodhpur domains and nested PDFs (153 URLs). It extracts raw HTML/PDF text, completely drops foreign-language noise via `langdetect`, cleans boilerplate artifacts (like "copyright" or "home"), and writes the strictly tokenized, lowercase output line-by-line exclusively into **`CleanedCorpus.txt`** (104,000+ tokens).
* **`problem1.py`**: Reads `CleanedCorpus.txt` to intrinsically train Continuous Bag-of-Words (CBOW) and Skip-Gram models exclusively utilizing `gensim.models.Word2Vec`. It iteratively constructs geometric topological maps testing combinations of Dimensions, Window Sizes, and Negative Samples. Finally, it validates performance via structural analogies (e.g., `student:exam :: faculty:?`), nearest-neighbor evaluations, and outputs semantic spacing plots using `sklearn`'s PCA and t-SNE algorithms.

### Sequence Generation (Problem 2)
* **`gen_names.py`**: A helper script utilized to synthesize a high-quality dataset of exactly 1,000 common Indian First/Last name combinations automatically structured into **`TrainingNames.txt`**.
* **`problem2.py`**: Reads the core `TrainingNames.txt` character representations tracking start-of-sequence, `<pad>`, and `<eos>`. It builds natively (from scratch via bare `torch.nn` layers) three recurrent architectures:
  1. **Vanilla RNN**
  2. **Bidirectional LSTM**
  3. **Vanilla RNN augmented with Self-Attention**
  
  The script trains the models recursively, and sequentially tests them by actively sampling strings at $T=0.8$. Finally, it mathematically evaluates **Diversity %** and **Novelty %** while logging hundreds of generated names to disk for qualitative manual inspection.



## ⚙️ Environment \& Setup

Ensure your Conda environment has the required pip packages running under `Python >= 3.8`.

```bash
# Activate your environment natively
source ~/.bashrc && conda activate my 

# Requirements
pip install beautifulsoup4 langdetect PyPDF2 requests wordcloud gensim scikit-learn matplotlib faker torch
```

---

## 🚀 Execution Guide

### 1. Rebuild the Corpus
If you need to pull fresh text embeddings mapping real-time academic policy from the university, execute the crawler:
```bash
python fetch_corpus.py
```
*(Warning: Scraping $\sim$153 nested dynamic URLs sequentially may take a few minutes. Final results stream into `CleanedCorpus.txt`)*

### 2. Run the Domain Embedding (Word2Vec) Task
Run the entire semantic geometry pipeline:
```bash
python problem1.py
```
**Expected Outputs:**
- `task1_wordcloud.png` (Corpus frequency visualization overriding trivial stop-words natively)
- `task4_clusters.png` (PCA Dimensional Space Map)
- `task4_tsne_clusters.png` (t-SNE Spatial Scatter)
- Prints real-time log matrix comparing Top-1 similarities for Word2Vec combinations. 

### 3. Run the Character Name Generator (RNNs) Tasks
First, seed the local names array natively:
```bash
python gen_names.py 
```
Subsequently, prompt the autoregressive pipeline:
```bash
python problem2.py
```
**Expected Outputs:**
- Direct graphical plot outputs of real-time Training Entropy/Cross-Entropy losses.
- Print statements mathematically calculating **Diversity** and **Novelty** indices per model loop.
- Creation of `GeneratedNames_VanillaRNN.txt`, `GeneratedNames_BLSTM.txt`, and `GeneratedNames_RNNAttention.txt` allowing manual sequential review of generated characters! 
