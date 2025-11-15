# 🧠 MLOps Project – nnc Classifier  
**Tác giả:** Nguyễn Nhất Chính  
**Trường:** FSB – Master of Software Engineering  
**Môn học:** MLOps  
**Deadline:** 17/11/2025  

---

## 🎯 Mục tiêu
Xây dựng một quy trình **MLOps đầy đủ** bao gồm:
1. Sinh dữ liệu huấn luyện mô phỏng bằng `make_classification`
2. Huấn luyện mô hình phân loại (RandomForest)
3. Ghi log quá trình bằng **MLflow Tracking**
4. Lưu và quản lý mô hình bằng **MLflow Model Registry**
5. Tạo ứng dụng web Flask sử dụng mô hình tốt nhất
6. Đóng gói toàn bộ ứng dụng bằng **Docker**
7. Thiết lập **CI/CD** tự động build & push image lên Docker Hub

---

## 🧩 Cấu trúc thư mục dự án

MLOps/
├── mlflow_project/
│ ├── train.py # Huấn luyện + log + đăng ký model
│ ├── tuning.py # Thử nghiệm tham số
│ ├── data_generator.py # Sinh dữ liệu mô phỏng
│ └── init.py
│
├── flask_app/
│ ├── app.py
│ ├── init.py
│ └── templates/
│ └── index.html
│
├── docker/
│ └── README.md
│
├── .github/
│ └── workflows/
│ └── docker-build.yml # CI/CD pipeline GitHub Actions
│
├── Dockerfile
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ Môi trường thực thi

| Thành phần | Phiên bản |
|-------------|------------|
| Python | 3.10.13 |
| MLflow | 2.14.1 |
| Flask | 3.0.3 |
| Scikit-learn | 1.4.2 |
| Numpy / Pandas | Mới nhất ổn định |
| OS | Windows 11 + PowerShell |
| IDE | Visual Studio Code |

---

## 🚀 Bước 1: Cài đặt môi trường

Tạo môi trường ảo:
```bash
python -m venv venv
venv\Scripts\activate
Nếu PowerShell báo lỗi:

bash
Copy code
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Cài đặt thư viện cần thiết:

bash
Copy code
pip install mlflow scikit-learn flask numpy pandas gunicorn matplotlib
📊 Bước 2: Huấn luyện mô hình và ghi log bằng MLflow
File: mlflow_project/train.py

Sử dụng make_classification() để sinh dữ liệu mẫu

Huấn luyện RandomForestClassifier

Ghi log tham số, metric (accuracy, f1_score)

Thực hiện 3 lần tuning → chọn model tốt nhất và đăng ký vào MLflow Model Registry

Chạy:

bash
Copy code
python mlflow_project/train.py
Ví dụ kết quả:

makefile
Copy code
n_estimators=50, max_depth=3, acc=0.8600, f1=0.8704  
n_estimators=100, max_depth=5, acc=0.8650, f1=0.8744  
n_estimators=150, max_depth=7, acc=0.8750, f1=0.8848  
✅ Best model logged & registered from run 545cfe034e9f4902944e82b745e5e7a7
Mở giao diện MLflow:

bash
Copy code
mlflow ui
→ Truy cập http://127.0.0.1:5000

🧠 Bước 3: Ứng dụng Flask cho mô hình tốt nhất
File: flask_app/app.py

Load model từ:

python
Copy code
mlflow.pyfunc.load_model("models:/nnc_classifier/1")
Giao diện web (flask_app/templates/index.html):
Form nhập 10 giá trị f1–f10 → trả về kết quả phân loại (0 hoặc 1).

Chạy Flask:

bash
Copy code
python flask_app/app.py
Truy cập: http://127.0.0.1:5000
Nếu thấy “Kết quả dự đoán: 1” → Flask App hoạt động thành công ✅

🐳 Bước 4: Đóng gói ứng dụng bằng Docker
File: Dockerfile

dockerfile
Copy code
# Base image Python nhẹ, ổn định
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file requirements.txt
COPY requirements.txt .

# Cài đặt thư viện
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn
COPY . .

# Mở cổng Flask
EXPOSE 5000

# Khởi chạy Flask app
CMD ["python", "flask_app/app.py"]
Build image:

bash
Copy code
docker build -t nnc-mlops .
Chạy container:

bash
Copy code
docker run -p 5000:5000 nnc-mlops
Truy cập:
👉 http://127.0.0.1:5000

⚙️ Bước 5: CI/CD Pipeline – GitHub Actions
Tự động build & push image lên Docker Hub mỗi khi có commit mới.

File: .github/workflows/docker-build.yml

yaml
Copy code
name: Build and Push Docker Image
on:
  push:
    branches: [ "main", "master" ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to DockerHub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_HUB_USERNAME }}
          password: ${{ secrets.DOCKER_HUB_PASSWORD }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ secrets.DOCKER_HUB_USERNAME }}/nnc-mlops:latest
✅ Khi push code → GitHub tự:

Build image

Login Docker Hub bằng token

Push image mới lên Docker Hub

Xem image tại:
🔗 https://hub.docker.com/repository/docker/kietlpa/nnc-mlops

🐋 Bước 6: Chạy image trực tiếp từ Docker Hub
Không cần build lại, chỉ cần:

bash
Copy code
docker pull kietlpa/nnc-mlops:latest
docker run -d -p 5000:5000 kietlpa/nnc-mlops:latest
Sau khi chạy:
→ Mở http://localhost:5000
→ Giao diện Flask hiển thị form nhập liệu và kết quả dự đoán.

☁️ Bước 7 (Bonus): Deploy lên Cloud
Có thể deploy trực tiếp image này lên:

Render.com

Railway.app

Azure Container App

Ví dụ:

bash
Copy code
docker run -d -p 80:5000 kietlpa/nnc-mlops:latest
Ứng dụng sau khi deploy có thể truy cập công khai qua domain cloud provider.

🧾 Tổng kết
Thành phần	Trạng thái	Ghi chú
MLflow Tracking	✅	Ghi log và quản lý model
Model Registry	✅	Lưu phiên bản model
Flask Web App	✅	Giao diện dự đoán
Dockerfile	✅	Build ổn định
CI/CD GitHub Actions	✅	Tự động push Docker Hub
Docker Hub Repo	✅	kietlpa/nnc-mlops
Bonus Cloud Deploy	🔜	Có thể triển khai thêm