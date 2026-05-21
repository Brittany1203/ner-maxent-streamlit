import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing import read_conll_file
from src.features import sentence_to_features
from src.model import load_model


TEST_PATH = "data/test.conll"
MODEL_PATH = "models/maxent_ner.pkl"
RESULTS_PATH = "results/evaluation_report.txt"


def main():
    sentences, labels = read_conll_file(TEST_PATH)
    model = load_model(MODEL_PATH)

    X_test = []
    y_true = []

    for tokens, sentence_labels in zip(sentences, labels):
        X_test.extend(sentence_to_features(tokens))
        y_true.extend(sentence_labels)

    y_pred = model.predict(X_test)

    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=["B-PER", "I-PER", "O"])

    print("Classification Report")
    print(report)

    print("Confusion Matrix")
    print(matrix)

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write("Classification Report\n")
        file.write(report)
        file.write("\n\nConfusion Matrix\n")
        file.write(str(matrix))

    df = pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred
    })
    df.to_csv("results/predictions.csv", index=False)

    print(f"Evaluation saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()