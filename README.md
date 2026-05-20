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

## 4. System Evaluation & Performance Metrics
To ensure the system's reliability and accuracy, we employ both qualitative and quantitative evaluation methods:

### Quantitative Metrics:
* **Retrieval Precision (Top-K Accuracy):** Measuring the percentage of relevant resumes successfully retrieved in the Top-K results for specific technical queries (e.g., querying "Python and Data Analysis" and verifying if the returned CVs contain these skills).
* **Latency Measurement:** Tracking the end-to-end response time from user query input in Streamlit to the generation of the final candidate list, ensuring the RAG pipeline operates efficiently.

### Qualitative Metrics (Manual Review):
* **Entity Extraction Accuracy:** Randomly sampling processed JSON outputs and manually comparing them against the original PDFs to ensure the LLM accurately captures names, skills, and correct domain classification without hallucinating.
* **Cross-Domain Retrieval (Outlier Handling):** Testing the system's ability to find "Banking candidates with IT skills" by explicitly filtering for the Banking domain while querying IT-specific keywords (e.g., Python, DevOps, Data pipelines) to validate the vector similarity logic.

## 5. Live Demo
* **App Link:** [[RAG-CV](https://rag-cv-kwzlvjtv4xak9dzubmgyky.streamlit.app/)]
