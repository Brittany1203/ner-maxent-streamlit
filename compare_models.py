import pandas as pd
from sklearn.metrics import classification_report

from src.preprocessing import read_conll_file
from src.features import sentence_to_features
from src.model import load_model
from baseline import build_person_name_dictionary, dictionary_baseline_predict


TEST_PATH = "data/test.conll"
MODEL_PATH = "models/maxent_ner.pkl"


def evaluate_model(y_true, y_pred, model_name):
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=["B-PER", "I-PER", "O"],
        output_dict=True,
        zero_division=0
    )

    return {
        "Model": model_name,
        "B-PER Precision": report_dict["B-PER"]["precision"],
        "B-PER Recall": report_dict["B-PER"]["recall"],
        "B-PER F1": report_dict["B-PER"]["f1-score"],
        "I-PER Precision": report_dict["I-PER"]["precision"],
        "I-PER Recall": report_dict["I-PER"]["recall"],
        "I-PER F1": report_dict["I-PER"]["f1-score"],
        "Accuracy": report_dict["accuracy"],
        "Macro F1": report_dict["macro avg"]["f1-score"],
        "Weighted F1": report_dict["weighted avg"]["f1-score"],
    }


def main():
    sentences, labels = read_conll_file(TEST_PATH)

    y_true = []
    for sentence_labels in labels:
        y_true.extend(sentence_labels)

    # Baseline model
    name_dictionary = build_person_name_dictionary()
    baseline_predictions = []

    for tokens in sentences:
        baseline_predictions.extend(
            dictionary_baseline_predict(tokens, name_dictionary)
        )

    # MaxEnt model
    maxent_model = load_model(MODEL_PATH)
    maxent_predictions = []

    for tokens in sentences:
        features = sentence_to_features(tokens)
        maxent_predictions.extend(maxent_model.predict(features))

    results = [
        evaluate_model(y_true, baseline_predictions, "Dictionary Baseline"),
        evaluate_model(y_true, maxent_predictions, "Maximum Entropy Classifier"),
    ]

    df = pd.DataFrame(results)
    df.to_csv("results/model_comparison.csv", index=False)

    print("\nModel Comparison")
    print(df.round(3))

    print("\nSaved comparison to results/model_comparison.csv")


if __name__ == "__main__":
    main()