# 📧 Spam Email Classifier (Naive Bayes)

## Overview
This project implements a spam email classifier using Natural Language Processing (NLP) and a Multinomial Naive Bayes model.  
It predicts the probability that an email is spam using both TF-IDF text features and engineered spam indicators.

## Features
- TF-IDF vectorization
- Custom spam indicators (exclamations, ALL CAPS, spam keywords)
- Probability-based spam prediction
- Model persistence with joblib
- Production-ready inference code

## Dataset
- 5,728 emails
- Labels: Spam (1) / Not Spam (0)

## Tech Stack
- Python
- NumPy, Pandas
- Scikit-learn
- SciPy
- Joblib

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

### 2. Run production inference
```bash
python src/production.py