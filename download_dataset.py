from datasets import load_dataset
import os

OUTPUT_DIR = "data"

NER_LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC"
]


def save_conll(split_data, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        for example in split_data:
            tokens = example["tokens"]
            ner_tags = example["ner_tags"]

            for token, tag_id in zip(tokens, ner_tags):
                label = NER_LABELS[tag_id]

                # This project focuses only on person-name recognition.
                # Other entity types are converted to O.
                if label not in {"B-PER", "I-PER"}:
                    label = "O"

                file.write(f"{token} {label}\n")

            file.write("\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_dataset("tomaarsen/conll2003")

    save_conll(dataset["train"], "data/train.conll")
    save_conll(dataset["validation"], "data/validation.conll")
    save_conll(dataset["test"], "data/test.conll")

    print("Dataset downloaded and converted successfully.")
    print("Saved files:")
    print("- data/train.conll")
    print("- data/validation.conll")
    print("- data/test.conll")


if __name__ == "__main__":
    main()