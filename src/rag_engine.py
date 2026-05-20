import os
import json
import glob
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

class RAGSystem:
    def __init__(self, persist_directory= os.path.join("data", "processed")):
        # Sử dụng model nhúng của OpenAI
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        self.persist_directory = persist_directory
        self.vector_store = None

    def load_data_and_build_index(self, processed_data_dir= os.path.join("data", "processed")):
        """Đọc file JSON đã trích xuất, gắn metadata và đưa vào Vector Database."""
        json_files = glob.glob(os.path.join(processed_data_dir, "*.json"))
        documents = []

        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Tạo text để nhúng
                content = f"Name: {data.get('candidate_name', '')}\n"
                content += f"Skills: {', '.join(data.get('skills', []))}\n"
                content += f"Experience: {data.get('experience_summary', '')}"
                
                # Gắn metadata để lọc theo yêu cầu đề bài
                metadata = {"domain": data.get('domain', 'UNKNOWN'), "source": file_path}
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

        print(f"[INFO] Đang chuẩn bị nhúng {len(documents)} CV vào Vector Database...")
        
        # Khởi tạo DB trống 
        self.vector_store = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embeddings
        )
        
        # Chia nhỏ dữ liệu, mỗi lần gửi 80 CV 
        batch_size = 80 
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            print(f"[INFO] Đang nhúng lô dữ liệu từ {i} đến {i + len(batch)}...")
            
            self.vector_store.add_documents(batch)
            
            # Nếu chưa phải lô cuối cùng thì cho hệ thống nghỉ 1 phút để hồi API quota
            if i + batch_size < len(documents):
                print("[INFO] Đã chạm ngưỡng an toàn. Hệ thống nghỉ 60 giây để tránh Rate Limit của Google...")
                time.sleep(60) 

        print("[SUCCESS] Đã xây dựng xong RAG Index cho toàn bộ CV!")
    
    def add_single_cv_to_index(self, file_path: str):
        """Chỉ nhúng và thêm DUY NHẤT 1 file CV JSON mới vào Vector DB hiện tại."""
        # Load database
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )
        
        # Đọc nội dung file JSON được chỉ định
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Tạo text cấu trúc để nhúng
            content = f"Name: {data.get('candidate_name', '')}\n"
            content += f"Skills: {', '.join(data.get('skills', []))}\n"
            content += f"Experience: {data.get('experience_summary', '')}"
            
            metadata = {"domain": data.get('domain', 'UNKNOWN'), "source": file_path}
            
            doc = Document(page_content=content, metadata=metadata)
            
            # Thêm document này vào Chroma
            self.vector_store.add_documents([doc])
            print(f"[SUCCESS] Đã nhúng thêm CV mới vào DB: {file_path}")

    def query_candidates(self, query: str, domain_filter: str = None, top_k: int = 5):
        """Truy vấn tìm kiếm ứng viên kèm bộ lọc."""
        # Load database nếu chưa có trên memory
        if not self.vector_store:
            self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        
        # Thiết lập bộ lọc metadata
        filter_dict = {"domain": domain_filter} if domain_filter else None
        
        print(f"[INFO] Đang tìm kiếm: '{query}' | Bộ lọc: {filter_dict}")
        results = self.vector_store.similarity_search(
            query=query,
            k=top_k,
            filter=filter_dict
        )
        return results

if __name__ == "__main__":
    rag = RAGSystem()
    
    rag.load_data_and_build_index()

    print("\n--- TEST: TÌM ỨNG VIÊN BANKING CHO PROJECT IT ---")
    outlier_results = rag.query_candidates(
        query="Software, programming, data analysis, IT skills, technology workflow",
        domain_filter="BANKING",
        top_k=5
    )
    
    for i, doc in enumerate(outlier_results):
        print(f"\nTop {i+1}:")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")













# import os
# import json
# import glob
# from langchain_core.documents import Document
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from pinecone import Pinecone
# from dotenv import load_dotenv

# load_dotenv()

# class RAGSystem:
#     def __init__(self):
#         # 1. Khởi tạo Embeddings 
#         self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        
#         # 2. Khởi tạo kết nối Pinecone
#         self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
#         self.index_name = "rag-cv" 
#         self.vector_store = None

#     def add_single_cv_to_index(self, file_path: str):
#         """Nhúng CV và đẩy thẳng lên Cloud Pinecone"""
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
            
#             content = f"Name: {data.get('candidate_name', '')}\n"
#             content += f"Skills: {', '.join(data.get('skills', []))}\n"
#             content += f"Experience: {data.get('experience_summary', '')}"
            
#             metadata = {"domain": data.get('domain', 'UNKNOWN'), "source": file_path}
#             doc = Document(page_content=content, metadata=metadata)
            
#             # Khởi tạo hoặc kết nối tới Cloud Vector DB và đẩy dữ liệu lên
#             self.vector_store = PineconeVectorStore.from_documents(
#                 documents=[doc],
#                 embedding=self.embeddings,
#                 index_name=self.index_name
#             )
#             print(f"[SUCCESS] Đã nhúng và đẩy CV lên Cloud DB: {file_path}")

#     def query_candidates(self, query: str, domain_filter: str = None, top_k: int = 5):
#         """Truy vấn tìm kiếm từ Cloud DB"""
#         # Kết nối tới index đang có trên Cloud
#         self.vector_store = PineconeVectorStore(
#             index_name=self.index_name, 
#             embedding=self.embeddings
#         )
        
#         filter_dict = {"domain": domain_filter} if domain_filter else None
        
#         results = self.vector_store.similarity_search(
#             query=query,
#             k=top_k,
#             filter=filter_dict
#         )
#         return results