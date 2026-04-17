import streamlit as st
import json
import torch
import gc
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -----------------------------------------------------
#  🎨 CẤU HÌNH GIAO DIỆN
# -----------------------------------------------------
st.set_page_config(
    page_title="Trợ lý Quy chế Sinh viên",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.chat-container {
    max-width: 820px;
    margin-left: auto;
    margin-right: auto;
}

[data-testid="stChatMessageContent"] {
    font-size: 17px;
    line-height: 1.5;
}

.user-msg {
    background-color: #DCF8C6 !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

.assistant-msg {
    background-color: #F1F0F0 !important;
    border-radius: 12px !important;
    padding: 10px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Trợ lý Quy chế Sinh viên – ĐH Khoa học Huế")


# -----------------------------------------------------
#  📂 1. TẢI TÀI NGUYÊN (CACHE)
# -----------------------------------------------------
@st.cache_resource
def load_resources():
    status = st.sidebar.empty()

    # --- Load dữ liệu chunking ---
    status.info("⏳ Đang load dữ liệu...")
    with open("Data/chunking_file.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # --- Embedding Model (E5-Large) ---
    status.info("⏳ Đang load Embedding Model (E5-Large)...")
    embed_model = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
    formatted = ["passage: " + c for c in raw_data]
    corpus_emb = embed_model.encode(formatted, normalize_embeddings=True, show_progress_bar=True)
    del formatted
    gc.collect()

    # --- BM25 (Keyword Search) ---
    status.info("⏳ Khởi tạo BM25...")
    tokenized = [doc.lower().split(" ") for doc in raw_data]
    bm25 = BM25Okapi(tokenized)
    del tokenized
    gc.collect()

    # --- Reranker ---
    status.info("⏳ Đang load Reranker (BGE)...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda")

    # --- LLM (Qwen2.5-7B, nén 4-bit) ---
    status.info("⏳ Đang load LLM Qwen2.5-7B (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    status.success("✅ Tải xong tất cả tài nguyên!")
    return raw_data, embed_model, corpus_emb, bm25, reranker, tokenizer, model


raw_data, embed_model, corpus_embeddings, bm25, reranker, tokenizer, model = load_resources()


# -----------------------------------------------------
#  🔍 2. HYBRID SEARCH
# -----------------------------------------------------
def hybrid_search(query: str, top_k: int = 3) -> list[str]:
    """
    Tìm kiếm kết hợp Vector Search + BM25, sau đó rerank để lấy top_k đoạn liên quan nhất.
    """
    # Vector Search
    query_vec = embed_model.encode(["query: " + query], normalize_embeddings=True)
    sim_scores = np.dot(corpus_embeddings, query_vec.T).flatten()
    vector_top_idx = np.argsort(sim_scores)[-12:][::-1]

    # BM25 Keyword Search
    tokenized_query = query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_idx = np.argsort(bm25_scores)[-12:][::-1]

    # Gộp kết quả ứng viên (loại trùng)
    combined_idx = list(set(list(vector_top_idx) + list(bm25_top_idx)))
    candidates = [raw_data[i] for i in combined_idx]

    if not candidates:
        return []

    # Reranking
    rerank_inputs = [[query, chunk] for chunk in candidates]
    scores = reranker.predict(rerank_inputs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [item[0] for item in ranked[:top_k]]


# -----------------------------------------------------
#  🤖 3. SINH CÂU TRẢ LỜI (RAG)
# -----------------------------------------------------
SYSTEM_PROMPT = """
Bạn là **Trợ lý ảo dành cho sinh viên Trường Đại học Khoa học – Đại học Huế**.

Nhiệm vụ của bạn:
1. Luôn trả lời dựa trên **"Thông tin tham khảo"** mà người dùng cung cấp ở mỗi câu hỏi.
2. Chỉ được sử dụng thông tin trong tài liệu đó.
3. Nếu người dùng hỏi điều KHÔNG có trong "Thông tin tham khảo", bạn phải trả lời:
   "Xin lỗi, thông tin này không nằm trong tài liệu người dùng đã cung cấp. Tôi không thể trả lời."
4. Tuyệt đối không tự bịa thông tin.
5. Nếu có nhiều đoạn tài liệu, bạn phải:
   - Đọc tất cả
   - Tổng hợp chính xác, dễ hiểu
   - Giữ ngữ nghĩa đúng với tài liệu gốc
6. Khi trả lời:
   - Viết rõ ràng, mạch lạc
   - Giải thích đơn giản, thân thiện
   - Không dùng thuật ngữ khó nếu không cần thiết
7. Nếu câu hỏi của người dùng không liên quan đến tài liệu (ví dụ: lập trình, đời sống…):
   → Trả lời: "Câu hỏi này nằm ngoài phạm vi Thông tin tham khảo."

Hãy luôn tuân thủ các quy tắc trên trong mọi tình huống.
"""

def generate_response(query: str) -> str:
    """
    Tìm ngữ cảnh bằng Hybrid Search, rồi dùng LLM sinh câu trả lời.
    """
    retrieved_chunks = hybrid_search(query, top_k=3)
    ctx = "\n\n".join(retrieved_chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Thông tin tham khảo:\n{ctx}\n\nCâu hỏi: {query}"},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=600,
        temperature=0.3,
        do_sample=True,
    )

    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer.split("assistant\n")[-1]


# -----------------------------------------------------
#  💬 4. GIAO DIỆN CHAT
# -----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
st.sidebar.header("⚙️ Tuỳ chọn")
if st.sidebar.button("🧹 Xóa lịch sử chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Mô hình sử dụng:**\n"
    "- 🔍 Embedding: `multilingual-e5-large`\n"
    "- 📋 Keyword: `BM25`\n"
    "- ⚖️ Reranker: `bge-reranker-v2-m3`\n"
    "- 🤖 LLM: `Qwen2.5-7B-Instruct` (4-bit)\n"
)

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi về quy chế sinh viên..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Đang tìm kiếm và tổng hợp câu trả lời..."):
            answer = generate_response(prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
