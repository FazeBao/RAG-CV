import fitz  # PyMuPDF
import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import os
import re

def extract_text_from_pdf(pdf_path: str) -> str:
    """Trích xuất text từ file PDF thông thường."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    
    # Các bước làm sạch text cơ bản
    text = text.replace('\xa0', ' ')  # Thay thế non-breaking space bằng khoảng trắng
    text = text.replace('Â', '')      # Loại bỏ ký tự Â bị lỗi
    text = re.sub(r'\n+', '\n', text) # Gom các dấu xuống dòng liên tiếp thành 1 dấu duy nhất
    return text.strip()

def preprocess_image_for_ocr(image):
    """Tiền xử lý ảnh bằng OpenCV để tăng độ chính xác khi chạy Tesseract."""
    # Chuyển ảnh PIL từ pdf2image sang format numpy array của OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Chuyển sang ảnh xám 
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Khử nhiễu để làm nổi bật chữ
    processed_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return processed_img

def extract_text_with_ocr(pdf_path: str) -> str:
    """Chuyển PDF Scan thành ảnh và dùng OCR để đọc chữ."""

    images = convert_from_path(pdf_path)
    full_text = ""
    
    for img in images:
        processed_img = preprocess_image_for_ocr(img)
        # Tesseract để lấy chữ
        text = pytesseract.image_to_string(processed_img, lang='eng')
        full_text += text + "\n"
        
    return full_text.strip()

def process_cv_document(pdf_path: str) -> str:
    """
    Kiểm tra định dạng PDF và trích xuất.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Không tìm thấy file: {pdf_path}")

    # 1. Thử đọc text bằng PyMuPDF trước
    text = extract_text_from_pdf(pdf_path)

    # 2. Nếu text quá ngắn (thường < 50-100 ký tự), rất có thể là Scanned PDF
    if len(text) < 100:
        print(f"[INFO] File {os.path.basename(pdf_path)} có vẻ là dạng Scan. Đang khởi động OCR pipeline...")
        text = extract_text_with_ocr(pdf_path)
    else:
        print(f"[INFO] Đọc thành công Text PDF: {os.path.basename(pdf_path)}.")

    return text

# Test
if __name__ == "__main__":
    # Test thử với 1 file trong thư mục raw của bạn
    # sample_path = r"C:\Project\RAG_CV_Project\RAG-CV\data\raw\BANKING\3547447.pdf" 
    # result = process_cv_document(sample_path)
    # print(result[:500]) # In 500 ký tự đầu tiên để kiểm tra
    pass