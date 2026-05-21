import streamlit as st
import re
import pandas as pd

st.set_page_config(
    page_title="NER MaxEnt Web App",
    page_icon="🔎",
    layout="centered"
)

st.title("Named Entity Recognition Web App")
st.write(
    "This app identifies possible person names in text. "
    "The current version is a rule-based baseline. "
    "It will later be upgraded to a Maximum Entropy classifier with custom linguistic features."
)

st.info(
    "Baseline rule: detect capitalized name-like tokens, excluding common sentence-start words and locations/organizations."
)

text = st.text_area(
    "Enter a sentence:",
    "Brittany and Aiko are doing great. Barack Obama met Angela Merkel in Berlin."
)

STOPWORDS = {
    "The", "A", "An", "This", "That", "These", "Those",
    "I", "You", "He", "She", "It", "We", "They",
    "In", "On", "At", "By", "For", "With", "From", "To",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "Berlin", "London", "Paris", "Macau", "Hong", "Kong", "China",
    "Apple", "Google", "Microsoft", "University"
}

def tokenize(text):
    return re.findall(r"\b[A-Za-z]+\b|[.,!?;]", text)

def is_name_like(token):
    return (
        re.match(r"^[A-Z][a-z]+$", token) is not None
        and token not in STOPWORDS
        and len(token) > 1
    )

def rule_based_ner(text):
    tokens = tokenize(text)
    predictions = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if is_name_like(token):
            name_tokens = [token]
            j = i + 1

            while j < len(tokens) and is_name_like(tokens[j]):
                name_tokens.append(tokens[j])
                j += 1

            full_name = " ".join(name_tokens)

            for k, name_token in enumerate(name_tokens):
                label = "B-PER" if k == 0 else "I-PER"
                predictions.append({
                    "Token": name_token,
                    "Label": label,
                    "Entity": full_name
                })

            i = j
        else:
            if re.match(r"^[A-Za-z]+$", token):
                predictions.append({
                    "Token": token,
                    "Label": "O",
                    "Entity": ""
                })
            i += 1

    return predictions

def extract_entities(predictions):
    entities = []
    current = []

    for row in predictions:
        if row["Label"] == "B-PER":
            if current:
                entities.append(" ".join(current))
            current = [row["Token"]]
        elif row["Label"] == "I-PER":
            current.append(row["Token"])
        else:
            if current:
                entities.append(" ".join(current))
                current = []

    if current:
        entities.append(" ".join(current))

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

if st.button("Detect Names"):
    predictions = rule_based_ner(text)
    entities = extract_entities(predictions)

    st.subheader("Detected Person Names")

    if entities:
        for entity in entities:
            st.success(entity)
    else:
        st.warning("No person names detected.")

    st.subheader("Highlighted Text")
    st.markdown(highlight_text(text, entities))

    st.subheader("Token-level BIO Predictions")
    df = pd.DataFrame(predictions)
    st.dataframe(df, use_container_width=True)

    st.subheader("Next Model Upgrade")
    st.write(
        "This rule-based system is used as a baseline. "
        "The next version will train a Maximum Entropy classifier using features such as "
        "capitalization, word shape, prefixes/suffixes, and context window."
    )