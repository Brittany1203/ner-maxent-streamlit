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