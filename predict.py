import re
from src.features import sentence_to_features
from src.model import load_model


MODEL_PATH = "models/maxent_ner.pkl"


def tokenize(text):
    return re.findall(r"\b[A-Za-z]+\b|[.,!?;]", text)


def predict_tokens(text):
    model = load_model(MODEL_PATH)
    tokens = tokenize(text)
    features = sentence_to_features(tokens)
    predictions = model.predict(features)

    return list(zip(tokens, predictions))


def extract_entities(token_predictions):
    entities = []
    current_entity = []

    for token, label in token_predictions:
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


if __name__ == "__main__":
    text = "Brittany met Tim Cook in Macau."
    predictions = predict_tokens(text)
    print(predictions)
    print(extract_entities(predictions))