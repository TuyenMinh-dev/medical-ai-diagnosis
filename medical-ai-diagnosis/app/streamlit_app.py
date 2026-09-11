"""
UI prototype đơn giản (tháng 7): upload ảnh y khoa -> gọi API backend (src/api/main.py) -> hiển thị kết quả.

Cách chạy:
    1. Chạy API backend trước:  uvicorn src.api.main:app --reload
    2. Chạy UI này ở terminal khác:  streamlit run app/streamlit_app.py
"""
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Medical AI Diagnosis - Demo", layout="centered")
st.title("Medical AI Diagnosis — Demo")
st.write("Tải lên 1 ảnh y khoa để mô hình dự đoán. (Chỉ để thử nghiệm, không dùng thay chẩn đoán y khoa thật.)")

uploaded_file = st.file_uploader("Chọn ảnh", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Ảnh đã tải lên", use_column_width=True)

    if st.button("Dự đoán"):
        with st.spinner("Đang gửi ảnh tới model..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files, timeout=30)
                response.raise_for_status()
                result = response.json()

                st.success(f"Kết quả: **{result['prediction']}** "
                           f"(độ tin cậy: {result['confidence']:.2%})")
                st.bar_chart(result["probabilities"])
            except requests.exceptions.ConnectionError:
                st.error("Không kết nối được tới API. Kiểm tra đã chạy "
                         "'uvicorn src.api.main:app --reload' chưa.")
            except Exception as e:
                st.error(f"Lỗi: {e}")
