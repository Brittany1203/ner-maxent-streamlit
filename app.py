import os
import re
import streamlit as st
import pandas as pd

from src.features import sentence_to_features
from src.model import load_model


MODEL_PATH = "models/maxent_ner.pkl"


st.set_page_config(
    page_title="NER MaxEnt Web App",
    page_icon="🔎",
    layout="centered"
)

st.title("Named Entity Recognition Web App")

st.write(
    "This app identifies person names using a Maximum Entropy-style classifier "
    "trained with custom linguistic features."
)

st.info(
    "The current model is trained on a small BIO-formatted sample dataset for demonstration. "
    "The pipeline is designed to support larger NER datasets."
)


def tokenize(text):
    return re.findall(r"\b[A-Za-z]+\b|[.,!?;]", text)


@st.cache_resource
def load_ner_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)


def predict_text(text, model):
    tokens = tokenize(text)
    features = sentence_to_features(tokens)
    predictions = model.predict(features)

    rows = []
    for token, label in zip(tokens, predictions):
        rows.append({
            "Token": token,
            "Predicted Label": label
        })

    return rows


def extract_entities(rows):
    entities = []
    current_entity = []

    for row in rows:
        token = row["Token"]
        label = row["Predicted Label"]

        if label == "B-PER":
            if current_entity:
                entities.append(" ".join(current_entity))
            current_entity = [token]
        elif label == "I-PER":
            if current_entity:
                current_entity.append(token)
        else:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []

    if current_entity:
        entities.append(" ".join(current_entity))

    return entities


def highlight_text(text, entities):
    highlighted = text
    for entity in sorted(entities, key=len, reverse=True):
        highlighted = re.sub(
            rf"\b{re.escape(entity)}\b",
            f":blue-background[{entity}]",
            highlighted
        )
    return highlighted


model = load_ner_model()

if model is None:
    st.error("Model file not found. Please run `python train.py` first.")
else:
    text = st.text_area(
        "Enter a sentence:",
        "Brittany met Tim Cook in Macau."
    )

    if st.button("Detect Names"):
        rows = predict_text(text, model)
        entities = extract_entities(rows)

        st.subheader("Detected Person Names")

        if entities:
            for entity in entities:
                st.success(entity)
        else:
            st.warning("No person names detected.")

        st.subheader("Highlighted Text")
        st.markdown(highlight_text(text, entities))

        st.subheader("Token-level BIO Predictions")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        st.subheader("Model Details")
        st.write(
            "Model: Maximum Entropy-style classifier using Logistic Regression "
            "with handcrafted token-level features."
        )
        st.write(
            "Features: lowercase word, word shape, capitalization, prefixes, suffixes, "
            "previous token features, next token features, and sentence boundary indicators."
        )