from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from preprocess import clean_text_batch

try:
    import xgboost as xgb

    XGB_OK = True
except ImportError:
    XGB_OK = False


def load_data(fake_csv: str, real_csv: str) -> tuple[pd.Series, pd.Series]:
    fake_df = pd.read_csv(fake_csv)
    real_df = pd.read_csv(real_csv)

    fake_df["label"] = 0
    real_df["label"] = 1

    if {"title", "text"}.issubset(fake_df.columns):
        fake_df["text"] = fake_df["title"].fillna("") + " " + fake_df["text"].fillna("")
    if {"title", "text"}.issubset(real_df.columns):
        real_df["text"] = real_df["title"].fillna("") + " " + real_df["text"].fillna("")

    data = pd.concat([fake_df[["text", "label"]], real_df[["text", "label"]]], ignore_index=True)
    data = data.dropna().drop_duplicates()

    return data["text"], data["label"]


def build_notebook_base_pipeline() -> Pipeline:
    estimators = [
        ("logreg", LogisticRegression(max_iter=500, C=1.0, solver="liblinear")),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=50,
                max_depth=4,
                random_state=42,
            ),
        ),
    ]

    if XGB_OK:
        estimators.append(
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=-1,
                ),
            )
        )

    voting = VotingClassifier(estimators=estimators, voting="soft")

    return Pipeline(
        steps=[
            ("cleaner", FunctionTransformer(clean_text_batch, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=3,
                    max_df=0.95,
                ),
            ),
            ("classifier", voting),
        ]
    )


def train_and_save(
    fake_csv: str,
    real_csv: str,
    output_dir: str = "models",
    sample_size: int = 5000,
) -> None:
    texts, labels = load_data(fake_csv, real_csv)

    if len(texts) > sample_size:
        sampled = pd.DataFrame({"text": texts, "label": labels})
        sampled = (
            sampled.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(n=sample_size // 2, random_state=42))
            .reset_index(drop=True)
        )
        texts = sampled["text"]
        labels = sampled["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model_pipeline = build_notebook_base_pipeline()
    model_pipeline.fit(x_train, y_train)

    preds = model_pipeline.predict(x_test)
    acc = accuracy_score(y_test, preds)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_pipeline, output_path / "prediction_model.pkl")

    print(f"Saved notebook-based model artifact to {output_path.resolve() / 'prediction_model.pkl'}")
    print(f"Validation accuracy: {acc:.4f}")


if __name__ == "__main__":
    # Update these paths based on your local dataset location.
    train_and_save("Fake.csv", "True.csv")
