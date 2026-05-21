from src.preprocessing import read_conll_file, flatten
from src.features import sentence_to_features
from src.model import build_maxent_model, save_model


TRAIN_PATH = "data/sample_train.conll"
MODEL_PATH = "models/maxent_ner.pkl"


def main():
    sentences, labels = read_conll_file(TRAIN_PATH)

    X = []
    y = []

    for tokens, sentence_labels in zip(sentences, labels):
        X.extend(sentence_to_features(tokens))
        y.extend(sentence_labels)

    model = build_maxent_model()
    model.fit(X, y)

    save_model(model, MODEL_PATH)

    print(f"Model trained successfully.")
    print(f"Training sentences: {len(sentences)}")
    print(f"Training tokens: {len(y)}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()