import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_maxent_model():
    model = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True)),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            multi_class="auto"
        ))
    ])

    return model


def save_model(model, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(model, file)


def load_model(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)