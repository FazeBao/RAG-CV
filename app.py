import os
import streamlit as st
from src.rag_engine import RAGSystem
from src.document_loader import process_cv_document
from src.llm_extractor import extract_entities_to_json

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="AI CV Search", page_icon="📄", layout="wide")

st.title("📄 AI Resume Retrieval System")
st.markdown("Search and filter top candidates using RAG and LLM.")

# Khởi tạo RAG 
@st.cache_resource
def init_rag_system():
    return RAGSystem()

rag = init_rag_system()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📥 1. Upload CV")
    uploaded_files = st.file_uploader("Upload candidates' resumes", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process & Upload"):
        if uploaded_files:
            with st.spinner("Đang xử lý CV và đưa vào cơ sở dữ liệu..."):
                raw_save_dir = os.path.join("data", "raw_uploaded")
                os.makedirs(raw_save_dir, exist_ok=True)
                
                for uploaded_file in uploaded_files:
                    try:
                        # 1. Lưu file
                        temp_pdf_path = os.path.join(raw_save_dir, uploaded_file.name)
                        with open(temp_pdf_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.write(f"🔄 Đang đọc file: {uploaded_file.name}...")
                        
                        # 2. Extract Text
                        cv_raw_text = process_cv_document(temp_pdf_path)
                        
                        # 3. Chạy LLM lấy JSON
                        output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_extracted.json"
                        processed_json_path = extract_entities_to_json(cv_raw_text, output_filename)
                        
                        # 4. CHỈ NHÚNG FILE VỪA TẠO
                        st.write(f"⚙️ Đang nhúng riêng ứng viên {uploaded_file.name} vào Vector DB...")
                        rag.add_single_cv_to_index(processed_json_path)
                        
                        st.success(f"✅ Đã trích xuất và nhúng xong: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ Lỗi khi xử lý file {uploaded_file.name}: {e}")
                
                st.success(f"🎉 Đã cập nhật thành công {len(uploaded_files)} ứng viên mới!")
        else:
            st.warning("Vui lòng chọn ít nhất 1 file để tải lên.")
            
    st.divider() 
    
    # BIẾN FILTER 
    st.header("🔍 2. Search Filters")
    domain_filter = st.selectbox(
        "Select Domain:",
        options=["ALL", "BANKING", "INFORMATION-TECHNOLOGY"]
    )
    top_k = st.slider("Number of Candidates (Top K):", min_value=1, max_value=10, value=5)


# --- GIAO DIỆN CHAT ---
# 1. Khởi tạo session_state để lưu trữ lịch sử
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI HR Assistant. How can I help you find candidates today?"}]

# 2. Hiển thị lại lịch sử chat 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Thanh Chat
if prompt := st.chat_input("Enter your requirements (e.g., 'Banking candidate with IT skills'):"):
    
    # Hiển thị câu hỏi 
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Hiển thị phản hồi 
    with st.chat_message("assistant"):
        with st.spinner("Searching database..."):
            
            # Xử lý logic filter 
            filter_val = None if domain_filter == "ALL" else domain_filter
            
            # Thực hiện truy vấn
            results = rag.query_candidates(
                query=prompt, 
                domain_filter=filter_val, 
                top_k=top_k
            )
            
            if results:
                summary_msg = f"✅ **Found {len(results)} matching candidates!** Here are the details:"
                st.markdown(summary_msg)
                
                for i, doc in enumerate(results):
                    with st.expander(f"Candidate #{i+1} - Domain: {doc.metadata.get('domain')}", expanded=True):
                        st.markdown(f"**Document Source:** `{doc.metadata.get('source')}`")
                        st.text(doc.page_content)
                
                # Lưu 
                st.session_state.messages.append({"role": "assistant", "content": summary_msg + " *(Details are expanded above)*"})
            
            else:
                error_msg = "⚠️ No candidates found matching your criteria. Please try different keywords."
                st.warning(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})