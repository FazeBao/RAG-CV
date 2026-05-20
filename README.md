# 📄 AI Resume Retrieval System

## 1. Overview
This project is an end-to-end Large Language Model (LLM) and Retrieval-Augmented Generation (RAG) tool designed to extract, structure, and search candidate information from unstructured resumes (CVs). It supports both digitally generated PDFs and image-based scanned PDFs, providing a lightweight chat interface for HR professionals to query and filter candidates effectively.

## 2. Technology Stack
* **Frontend & Deployment:** Streamlit, deployed on Streamlit Community Cloud.
* **Document Processing & OCR:** PyMuPDF (`fitz`), OpenCV, PyTesseract (for scanned documents).
* **LLM & Orchestration:** LangChain, Google Gemini 3.1 Flash Lite (for structured entity extraction), Gemini Embedding 001.
* **Vector Database:** ChromaDB (Local vector storage).
* **Data Format:** JSON (for storing extracted entities).

## 3. System Architecture & Workflow
The system is modularized into four main components:

1.  **Document Ingestion (`src/document_loader.py`):** Reads uploaded PDF files. It uses PyMuPDF for standard digital text extraction. If the text length is unusually short (indicating a scanned PDF), it automatically triggers an OCR pipeline using OpenCV for preprocessing and Tesseract for text extraction.
2.  **LLM Entity Extraction (`src/llm_extractor.py`):** Passes the raw text to the Gemini LLM using LangChain's structured output. The prompt enforces the extraction of specific entities: `candidate_name`, `skills`, `experience_summary`, and `domain` (BANKING or INFORMATION-TECHNOLOGY). The output is saved locally as a JSON file.
3.  **RAG Engine (`src/rag_engine.py`):** Reads the structured JSON files, formats them into searchable textual content, attaches metadata (like domain and source file), and converts them into embeddings using Gemini Embeddings. These vectors are indexed and stored in ChromaDB for fast similarity search and metadata filtering.
4.  **User Interface & Chat (`app.py`):** A Streamlit application allowing users to upload new CVs, process them on the fly, and use a chat interface to query candidates. It includes metadata filtering capabilities to seamlessly isolate or cross-reference specific domains.

## 4. Known Limitations
* **Empty Initial Database:** Due to file size and storage constraints on GitHub and Streamlit Community Cloud, the complete dataset and pre-built vector database cannot be hosted. As a result, the database will be completely empty when you first access the live Streamlit app.
* **Testing the App:** To evaluate the system's capabilities, users are encouraged to manually upload a few sample CVs using the upload feature on the sidebar. 
* **Ephemeral Storage:** Streamlit Community Cloud instances go to sleep after a period of inactivity. Please note that once the application restarts or goes to sleep, any previously uploaded CVs and their generated data will be permanently deleted.

## 5. Live Demo
* **App Link:** [[RAG-CV](https://rag-cv-kwzlvjtv4xak9dzubmgyky.streamlit.app/)]
