from src.preprocessing import read_conll_file


TRAIN_PATH = "data/train.conll"


def build_person_name_dictionary(train_path=TRAIN_PATH):
    sentences, labels = read_conll_file(train_path)

    name_tokens = set()

    for tokens, sentence_labels in zip(sentences, labels):
        for token, label in zip(tokens, sentence_labels):
            if label in {"B-PER", "I-PER"}:
                name_tokens.add(token.lower())

    return name_tokens


def dictionary_baseline_predict(tokens, name_dictionary):
    predictions = []

    previous_is_person = False

    for token in tokens:
        if token.lower() in name_dictionary:
            if previous_is_person:
                predictions.append("I-PER")
            else:
                predictions.append("B-PER")
            previous_is_person = True
        else:
            predictions.append("O")
            previous_is_person = False

    return predictions