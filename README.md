# NLP Named Entity Recognition Web App

This project builds a person-name recognition system using a Maximum Entropy-style classifier with custom linguistic features. The model is trained on a CoNLL-style named entity recognition dataset and deployed through an interactive Streamlit web app.

## Project Overview

Named Entity Recognition is a core Natural Language Processing task that identifies entities such as person names, organizations, and locations from text.

This project focuses on person-name recognition using BIO tagging:

- B-PER: beginning of a person name
- I-PER: inside a person name
- O: outside a person name

## Methods

- Token-level classification
- Maximum Entropy-style classifier using Logistic Regression
- Custom linguistic feature engineering
- Dictionary-based baseline comparison
- Precision, recall, F1-score, and confusion matrix evaluation
- Streamlit web app deployment

## Feature Engineering

The model uses handcrafted linguistic and contextual features, including:

- Lowercase word
- Word shape
- Capitalization
- Prefixes and suffixes
- Previous token features
- Next token features
- Sentence boundary indicators

## Dataset

This project uses a CoNLL-style named entity recognition dataset converted into BIO format. Since the project focuses on person-name recognition, non-person entity labels such as ORG, LOC, and MISC are converted to `O`.

The dataset includes:

- 14,041 training sentences
- 203,621 training tokens
- A separate test set for model evaluation

A small sample dataset is also included for debugging and reproducibility.

## Evaluation Results

| Label | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| B-PER | 0.91 | 0.75 | 0.82 | 1617 |
| I-PER | 0.89 | 0.88 | 0.88 | 1156 |
| O | 0.99 | 0.99 | 0.99 | 43662 |

Overall accuracy: **0.98**

## Baseline Comparison

To evaluate whether the Maximum Entropy classifier improves over a simple rule-based approach, I compared it with a dictionary-based baseline.

The dictionary baseline extracts person-name tokens from the training set and predicts a token as `B-PER` or `I-PER` if it appears in the learned name dictionary.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Dictionary Baseline | 0.851 | 0.476 | 0.879 |
| Maximum Entropy Classifier | 0.983 | 0.899 | 0.983 |

The Maximum Entropy classifier performs better because it uses contextual and linguistic features rather than relying only on whether a token has appeared as a name in the training data.

## Streamlit Demo

The Streamlit app allows users to enter a sentence and view:

- Detected person names
- Highlighted text
- Token-level BIO predictions
- Model details

## Limitations

- The current model focuses only on person-name recognition.
- Non-person entities are treated as outside labels.
- The model relies on handcrafted linguistic features rather than contextual embeddings.
- The model may miss rare or unseen single-token names, such as names that do not appear frequently in the training dataset.
- Person names with both first and last names are generally easier to detect than isolated first names.
- Future improvements could train a full multi-class NER model for PER, ORG, LOC, and MISC instead of converting non-person labels to `O`.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt

