# ---------------------------
# Stage 1: Build Flask + MLflow App
# ---------------------------

# Base image Python khớp môi trường huấn luyện
FROM python:3.11-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Đặt MLflow dùng tracking nội bộ
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Sao chép file requirements.txt vào container
COPY requirements.txt .

# Cài đặt thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Expose cổng Flask
EXPOSE 5000

# Lệnh khởi chạy Flask app
CMD ["python", "flask_app/app.py"]
