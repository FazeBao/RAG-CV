import json
import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()

# 1. Ép LLM trả về đúng định dạng 
class CVInfo(BaseModel):
    candidate_name: str = Field(description="Tên của ứng viên. Nếu không tìm thấy trong văn bản, ghi 'Unknown'")
    skills: list[str] = Field(description="Danh sách các kỹ năng chuyên môn, công nghệ hoặc nghiệp vụ của ứng viên")
    experience_summary: str = Field(description="Tóm tắt ngắn gọn kinh nghiệm làm việc nổi bật nhất (dưới 50 từ)")
    domain: str = Field(description="Phân loại lĩnh vực: Chỉ được phép trả về 'BANKING' hoặc 'INFORMATION-TECHNOLOGY'")

def extract_entities_to_json(cv_text: str, output_filename: str) -> str:
    """
    Sử dụng LLM trích xuất thông tin từ CV và lưu thành file JSON.
    """
    # 2. Khởi tạo LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0) # Temperature = 0 để tăng tính chính xác
    
    structured_llm = llm.with_structured_output(CVInfo)
    
    # 3. Thiết kế System Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là một chuyên gia nhân sự và trích xuất dữ liệu xuất sắc. 
        Nhiệm vụ của bạn là đọc nội dung CV và trích xuất các thông tin chính xác theo cấu trúc được yêu cầu.
        Không bịa đặt thông tin nếu CV không đề cập."""),
        ("human", "Dưới đây là nội dung CV thô:\n\n{cv_text}\n\nHãy tiến hành trích xuất.")
    ])
    
    # 4. Pipeline thực thi
    chain = prompt | structured_llm
    print("[INFO] Đang gửi yêu cầu đến LLM để trích xuất dữ liệu...")
    result = chain.invoke({"cv_text": cv_text})
    
    # 5. Lưu vào file JSON trong thư mục data/processed
    base_output_dir = os.path.join("data", "processed")
    output_path = os.path.join(base_output_dir, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=4)
        
    print(f"[SUCCESS] Đã lưu thông tin trích xuất vào: {output_path}")
    return output_path

# Test
if __name__ == "__main__":
    import glob
    from document_loader import process_cv_document
    import time

    raw_data_dirs = [os.path.join("data", "raw", "BANKING"), os.path.join("data", "raw", "INFORMATION-TECHNOLOGY")]
    for data_dir in raw_data_dirs:
        # Lấy danh sách tất cả các file PDF trong thư mục
        pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
        
        for pdf_path in pdf_files:
            try:
                print(f"\n--- Đang xử lý: {os.path.basename(pdf_path)} ---")
                # 1. Trích xuất text
                cv_raw_text = process_cv_document(pdf_path)
                
                # 2. Đặt tên file JSON đầu ra tương ứng
                filename_without_ext = os.path.splitext(os.path.basename(pdf_path))[0]
                output_filename = f"{filename_without_ext}_extracted.json"
                
                # 3. Trích xuất bằng LLM
                extract_entities_to_json(cv_raw_text, output_filename)
                
                time.sleep(2)
                
            except Exception as e:
                print(f"[ERROR] Lỗi khi xử lý file {pdf_path}: {e}")
                
    print("\n[DONE] Đã hoàn tất xử lý hàng loạt toàn bộ CV!")