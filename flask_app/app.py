from flask import Flask, render_template, request
import mlflow.pyfunc
import numpy as np
import os
from pathlib import Path

# Tên model trong MLflow Registry
MODEL_NAME = os.getenv("MODEL_NAME", "nnc_classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"   # hoặc "models:/nnc_classifier/Production" nếu bạn đặt stage
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")  # local | registry | auto


def _extract_value_from_meta(meta_path, key):
    """Đọc nhanh giá trị YAML mà không cần thêm thư viện."""
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{key}:"):
                return line.split(":", 1)[1].strip()
    return None


def _resolve_local_model_path(model_name, version):
    """Tìm đường dẫn artifact trong thư mục mlruns để dùng trong Docker."""
    base_dir = Path(__file__).resolve().parents[1] / "mlruns"
    version_meta = base_dir / "models" / model_name / f"version-{version}" / "meta.yaml"
    if not version_meta.exists():
        raise FileNotFoundError(f"Không tìm thấy meta cho model {model_name} version {version}")

    model_id = _extract_value_from_meta(version_meta, "model_id")
    if not model_id:
        raise ValueError("Không đọc được model_id từ meta.yaml")

    # Tìm artifacts tương ứng trong tất cả experiment
    for exp_dir in base_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "models":
            continue
        artifacts_dir = exp_dir / "models" / model_id / "artifacts"
        if artifacts_dir.exists():
            return artifacts_dir

    raise FileNotFoundError(f"Không tìm thấy artifacts cho model_id {model_id}")


def _load_from_local():
    local_path = _resolve_local_model_path(MODEL_NAME, MODEL_VERSION)
    print(f"➡️  Load model từ artifact cục bộ: {local_path}")
    return mlflow.pyfunc.load_model(str(local_path))


def _load_from_registry():
    print(f"➡️  Load model từ Registry: {MODEL_URI}")
    return mlflow.pyfunc.load_model(MODEL_URI)


def _load_model():
    loaders = []

    if MODEL_SOURCE == "local":
        loaders = [_load_from_local]
    elif MODEL_SOURCE == "registry":
        loaders = [_load_from_registry]
    else:  # auto
        loaders = [_load_from_local, _load_from_registry]

    last_err = None
    for loader in loaders:
        try:
            return loader()
        except Exception as err:
            print(f"⚠️  Không thể {loader.__name__}: {err}")
            last_err = err

    print(f"⛔ Không thể load model bằng bất kỳ nguồn nào: {last_err}")
    return None


app = Flask(__name__)
model = _load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Đọc 10 feature đầu vào
            features = [float(request.form[f"f{i}"]) for i in range(1, 11)]
            arr = np.array(features).reshape(1, -1)
            prediction = int(model.predict(arr)[0])
        except Exception as e:
            prediction = f"Lỗi khi dự đoán: {e}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
