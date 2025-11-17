from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from mlflow_project.data_generator import generate_classification_data

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "classification_data.csv"

mlflow.set_tracking_uri("file:./mlruns")  # Lưu log cục bộ
mlflow.set_experiment("nnc_classification")

def train_and_log_model(n_estimators, max_depth):
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"[LOAD] Loaded existing dataset from {DATA_PATH}")
    else:
        df, saved = generate_classification_data(save_path=DATA_PATH)
        print(f"[NEW] Generated dataset and saved to {saved}")

    X = df.drop(columns=["target"]).values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "nnc_model")

        print(f"n_estimators={n_estimators}, max_depth={max_depth}, acc={acc:.4f}, f1={f1:.4f}")
        return acc, f1, mlflow.active_run().info.run_id


if __name__ == "__main__":
    results = []
    # Thử nghiệm 3 lần (tuning)
    for n, d in [(50, 3), (100, 5), (150, 7)]:
        acc, f1, run_id = train_and_log_model(n, d)
        results.append((acc, f1, run_id))

    # Chọn model tốt nhất
    best = max(results, key=lambda x: x[0])
    best_run = best[2]

    # Đăng ký model tốt nhất vào Registry
    mlflow.register_model(
        f"runs:/{best_run}/nnc_model",
        "nnc_classifier"
    )

    print(f"[BEST] Model logged & registered from run {best_run}")
