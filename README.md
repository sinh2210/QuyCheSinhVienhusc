# 🎓 Trợ lý Quy chế Sinh viên – ĐH Khoa học Huế

Chatbot hỏi đáp về **Quy chế Sinh viên** của Trường Đại học Khoa học – Đại học Huế, sử dụng kỹ thuật **RAG Hybrid** kết hợp:

- 🔍 **Vector Search** (Embedding E5-Large) – tìm kiếm theo ngữ nghĩa
- 📋 **BM25** – tìm kiếm theo từ khóa
- ⚖️ **Reranker** (BGE-reranker-v2-m3) – chấm điểm và chọn đoạn liên quan nhất
- 🤖 **LLM** (Qwen2.5-7B-Instruct, nén 4-bit) – sinh câu trả lời

---

## 📁 Cấu trúc dự án

```
QuyCheSinhVienhusc/
├── app.py                  # Ứng dụng Streamlit chính
├── run_colab.py            # Script deploy lên Colab qua ngrok
├── requirements.txt        # Danh sách thư viện
├── token.txt               # Ngrok token (KHÔNG đẩy lên GitHub!)
├── Data/
│   └── chunking_file.json  # Dữ liệu quy chế đã chunk sẵn
└── QuyCheSinhVien.ipynb    # Notebook gốc (thử nghiệm)
```

---

## 🚀 Hướng dẫn chạy trên Google Colab

### Bước 1 – Cài thư viện

```python
!pip install -q -U torch transformers sentence-transformers bitsandbytes accelerate rank_bm25 streamlit pyngrok
```

### Bước 2 – Upload các file cần thiết lên Colab

- `app.py`
- `Data/chunking_file.json`
- `token.txt` (chứa ngrok authtoken, lấy tại [ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken))

### Bước 3 – Chạy script deploy

```python
!python run_colab.py
```

Sau khi chạy xong, bạn sẽ thấy link dạng `https://xxxx.ngrok-free.app` – mở link đó để dùng chatbot.

---

## ⚠️ Lưu ý

- Cần GPU (Google Colab T4 hoặc tốt hơn) để chạy mô hình 7B.
- File `token.txt` chứa token riêng tư – **không đẩy lên GitHub**. Đã liệt kê trong `.gitignore`.
- Model LLM sẽ được tải tự động từ HuggingFace lần đầu chạy (~14GB).

---

## 📦 Model sẵn trên Google Drive

Nếu không muốn tải lại: [Link Drive](https://drive.google.com/drive/folders/151NSXergkOFiUXm0Z6Nna9jjzlGeBxri)  
Tắt tắt vào Drive của bạn rồi trỏ đường dẫn trong `app.py` là được.
