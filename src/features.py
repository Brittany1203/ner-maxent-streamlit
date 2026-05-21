def word_shape(word):
    shape = ""
    for char in word:
        if char.isupper():
            shape += "X"
        elif char.islower():
            shape += "x"
        elif char.isdigit():
            shape += "d"
        else:
            shape += char
    return shape


def extract_features(tokens, i):
    word = tokens[i]

    features = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word.shape": word_shape(word),
        "word.istitle": word.istitle(),
        "word.isupper": word.isupper(),
        "word.isdigit": word.isdigit(),
        "prefix_1": word[:1],
        "prefix_2": word[:2],
        "prefix_3": word[:3],
        "suffix_1": word[-1:],
        "suffix_2": word[-2:],
        "suffix_3": word[-3:],
    }

    if i > 0:
        prev_word = tokens[i - 1]
        features.update({
            "prev_word.lower": prev_word.lower(),
            "prev_word.istitle": prev_word.istitle(),
            "prev_word.shape": word_shape(prev_word),
        })
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        next_word = tokens[i + 1]
        features.update({
            "next_word.lower": next_word.lower(),
            "next_word.istitle": next_word.istitle(),
            "next_word.shape": word_shape(next_word),
        })
    else:
        features["EOS"] = True

    return features


def sentence_to_features(tokens):
    return [extract_features(tokens, i) for i in range(len(tokens))]