# =============================================================
#  Script chạy trên Google Colab để deploy Streamlit qua ngrok
#  Bước 1: Kết nối Google Drive (chứa model đã tải sẵn)
#  Bước 2: Đọc ngrok token từ file token.txt
#  Bước 3: Khởi động Streamlit và expose qua ngrok
# =============================================================
#
#  📦 Model đã tải sẵn trên Drive:
#  https://drive.google.com/drive/folders/151NSXergkOFiUXm0Z6Nna9jjzlGeBxri
#
#  🔑 Tạo ngrok token tại: https://dashboard.ngrok.com/get-started/your-authtoken
#  Lưu token vào file token.txt rồi upload lên Colab.
# =============================================================

import os
import subprocess
from pyngrok import ngrok

# --- Bước 1: Kết nối Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Bước 2: Đọc ngrok token ---
with open("token.txt", "r") as f:
    NGROK_TOKEN = f.read().strip()

print("🔑 Token đã được load từ file.")

# --- Bước 3: Cấu hình ngrok ---
ngrok.kill()  # Tắt kết nối cũ nếu có
os.system(f"ngrok config add-authtoken {NGROK_TOKEN}")

# Mở cổng 8501 (cổng mặc định của Streamlit)
public_url = ngrok.connect(8501).public_url
print(f"\n🚀 LINK CHATBOT CỦA BẠN:\n{public_url}\n")

# --- Bước 4: Chạy Streamlit ---
os.system("streamlit run app.py")
