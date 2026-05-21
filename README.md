# NLP Named Entity Recognition Web App

This project builds a Named Entity Recognition system for detecting person names using a Maximum Entropy classifier and deploys it as an interactive Streamlit web application.

## Project Overview

Named Entity Recognition is a core Natural Language Processing task that identifies named entities such as person names, organizations, and locations from text.

In this project, I focus on person-name recognition using BIO tagging:

- B-PER: beginning of a person name
- I-PER: inside a person name
- O: outside a person name

## Methods

- Token-level classification
- Maximum Entropy classifier
- Custom linguistic feature engineering
- Precision, recall, and F1-score evaluation
- Streamlit web app deployment

## Features

The model uses linguistic and contextual features including:

- Word shape
- Capitalization
- Prefixes and suffixes
- Previous and next words
- Sentence boundary indicators

## Tech Stack

Python, scikit-learn, pandas, Streamlit, NLTK

## Dataset

This project uses a CoNLL-style named entity recognition dataset converted into BIO format. Since the project focuses on person-name recognition, non-person entity labels such as ORG, LOC, and MISC are converted to `O`.

The dataset includes:

- 14,041 training sentences
- 203,621 training tokens
- A separate test set for model evaluation

A small sample dataset is also included for debugging and reproducibility.

## Evaluation Results

The Maximum Entropy-style classifier was evaluated using precision, recall, and F1-score.

| Label | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| B-PER | 0.91 | 0.75 | 0.82 | 1617 |
| I-PER | 0.89 | 0.88 | 0.88 | 1156 |
| O | 0.99 | 0.99 | 0.99 | 43662 |

Overall accuracy: **0.98**

## Model Interpretation

The model performs well at identifying person-name tokens, especially inside-name tokens (`I-PER`). The lower recall for `B-PER` suggests that the model sometimes misses the beginning of person names, which is a common challenge in token-level NER.

## Limitations

- The current model focuses only on person-name recognition.
- Non-person entities are treated as outside labels.
- The model relies on handcrafted linguistic features rather than contextual embeddings.
- Future improvements could include CRF, BiLSTM, or transformer-based models such as BERT.

## Baseline Comparison

To evaluate whether the Maximum Entropy classifier improves over a simple rule-based approach, I compared it with a dictionary-based baseline.

The dictionary baseline extracts person-name tokens from the training set and predicts a token as `B-PER` or `I-PER` if it appears in the learned name dictionary.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Dictionary Baseline | 0.851 | 0.476 | 0.879 |
| Maximum Entropy Classifier | 0.983 | 0.899 | 0.983 |

The Maximum Entropy classifier performs better because it uses contextual and linguistic features rather than relying only on whether a token has appeared as a name in the training data.