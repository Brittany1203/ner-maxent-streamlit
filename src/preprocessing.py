def read_conll_file(file_path):
    sentences = []
    labels = []

    current_tokens = []
    current_labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
                continue

            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                label = parts[-1]
                current_tokens.append(token)
                current_labels.append(label)

    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)

    return sentences, labels


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]