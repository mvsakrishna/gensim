# Gensim NLP Project

## Overview
This repository contains code and examples for using **Gensim**, a powerful Python library for topic modeling, document similarity, and word embeddings. The project demonstrates how to process text data, train models, and extract meaningful insights using NLP techniques.

## Features
- Text preprocessing (tokenization, stopword removal, lemmatization)
- Training word embeddings (Word2Vec, FastText, GloVe)
- Topic modeling with **Latent Dirichlet Allocation (LDA)**
- Document similarity and retrieval
- Evaluation and visualization of NLP models

## Installation

Clone the repository:
```bash
git clone https://github.com/your-username/gensim-nlp-project.git
cd gensim-nlp-project
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies
Ensure you have the following installed:
```txt
gensim
numpy
pandas
nltk
scikit-learn
matplotlib
```

## Usage
### 1. Preprocess Text Data
Run the script to clean and preprocess text data:
```bash
python preprocess.py --input data/text_corpus.txt --output data/cleaned_texts.pkl
```

### 2. Train Word Embeddings
Train a **Word2Vec** model using:
```bash
python train_word2vec.py --input data/cleaned_texts.pkl --output models/word2vec.model
```

### 3. Topic Modeling with LDA
Generate topics from a corpus using LDA:
```bash
python train_lda.py --input data/cleaned_texts.pkl --num_topics 10 --output models/lda_model
```

### 4. Find Similar Documents
To find similar documents in a dataset:
```bash
python find_similar.py --query "sample query text" --model models/word2vec.model
```




