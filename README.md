# RAG Pipeline - Retrieval-Augmented Generation

A powerful Retrieval-Augmented Generation (RAG) system built with LangChain, Streamlit, and Chroma that enables intelligent document processing and question-answering over your PDF documents.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Screenshots](#project-screenshots)
- [Video Demo](#video-demo)
- [Folder Structure](#folder-structure)
- [How RAG Works](#how-rag-works)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Technologies Used](#technologies-used)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This RAG (Retrieval-Augmented Generation) pipeline allows you to:

- Upload and process PDF documents
- Create vector embeddings from document chunks
- Retrieve relevant documents based on queries
- Generate intelligent answers using LLMs (OpenAI, Groq)
- Interact with a user-friendly Streamlit web interface

The system combines retrieval and generation to provide accurate, context-aware responses based on your document corpus.

---

## ✨ Features

- **Multi-format Document Loading**: Support for PDFs, TXT, CSV, DOCX, and Excel files
- **Smart Document Chunking**: Configurable chunk size and overlap
- **Vector Embeddings**: Uses SentenceTransformers for semantic understanding
- **Vector Store**: Chroma database for efficient similarity search
- **LLM Integration**: Support for OpenAI and Groq models
- **Web Interface**: Interactive Streamlit dashboard
- **Configurable Pipeline**: Adjust embedding models, chunk parameters, and similarity thresholds
- **Reflection & Scoring**: Built-in answer quality scoring and reflection

---

## 📸 Project Screenshots

<!-- Project screenshots (local assets) -->

### Screenshot 1: Main Dashboard

![Main Dashboard](assets/RAG%201.png)

### Screenshot 2: Configuration Panel

![Configuration Panel](assets/RAG%202.png)

### Screenshot 3: Document Retrieval Results

![Retrieval Results](assets/RAG%203.png)

### Screenshot 4: Additional View

![Additional View](assets/RAG%204.png)

---

### Video Demo

Below is a demo video demonstrating the main flows (indexing, retrieval, and answering). Use the link below to view it.

- **Google Drive (direct view):** https://drive.google.com/file/d/1nLbFjJuFEJ7dbytmqJH4bpA-Ac6lMgI6/view?usp=sharing


**What the demo shows:**

- Uploading PDF documents
- Indexing and processing documents
- Querying the system
- Viewing retrieved documents and answers

---

## 📁 Folder Structure

```
RAG Task/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration
├── README.md                       # This file
│
├──assets/                          # Project assets (images, videos)
|
├──.env                             # Environment variables (API keys)
|
├──main.py                          # Another main script (single program)
|
├── data/
│   ├── pdf/                        # Store your PDF files here
│   │   └── (your documents)
│   └── vector_store/               # Chroma vector database
│       ├── chroma.sqlite3          # Vector store database
│       └── [uuid-folders]/         # Embeddings storage
│
└── src/                            # Source code modules
    ├── __init__.py
    ├── loaders.py                  # Document loading logic
    ├── splitter.py                 # Text splitting and chunking
    ├── embeddings.py               # Embedding management
    ├── vectorstore.py              # Vector store operations
    ├── retriever.py                # RAG retrieval logic
    ├── llm.py                      # LLM integration
    └── __pycache__/                # Python cache

```

### Folder Descriptions

| Folder                 | Purpose                                                                       |
| ---------------------- | ----------------------------------------------------------------------------- |
| **data/pdf/**          | Upload your PDF documents here. Supported formats: PDF, TXT, CSV, DOCX, Excel |
| **data/vector_store/** | Automatically created storage for vector embeddings and metadata              |
| **src/**               | Core modules implementing the RAG pipeline                                    |

---

## 🧠 How RAG Works

### RAG Pipeline Overview

```
          User Query
              ↓
[1] Embedding Generation
    • Convert query to vector embedding
    • Use SentenceTransformers model
              ↓
[2] Vector Similarity Search (Retrieval)
    • Search Chroma vector database
    • Retrieve top-K most similar documents
    • Filter by similarity threshold
              ↓
[3] Context Assembly
    • Combine retrieved documents
    • Prepare context for LLM
              ↓
[4] LLM Answer Generation
    • Send query + context to LLM (OpenAI/Groq)
    • Generate intelligent answer
              ↓
[5] Answer Reflection (Optional)
    • Score answer quality
    • Provide confidence metrics
              ↓
      Final Answer to User
```

### Detailed Process

#### 1. **Document Ingestion & Chunking**

- Load documents from `data/pdf/` folder
- Split documents into chunks (default: 1000 tokens)
- Maintain overlap between chunks (default: 200 tokens) for context continuity

#### 2. **Embedding Generation**

- Use SentenceTransformers (`all-MiniLM-L6-v2` by default)
- Convert text chunks into 384-dimensional vectors
- Store embeddings in Chroma vector database

#### 3. **Retrieval**

- User submits a query through the Streamlit interface
- Query is converted to the same vector embedding space
- Chroma performs similarity search using cosine distance
- Top-K most relevant documents are returned (configurable, default: 3)

#### 4. **Generation**

- Retrieved documents become context for the LLM
- Query + context is sent to LLM (OpenAI GPT or Groq Llama)
- LLM generates an answer grounded in the retrieved documents

#### 5. **Reflection** (Optional)

- System can reflect on answer quality
- Provides confidence scores and improvement suggestions

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager
- API keys (optional):
  - OpenAI API key (for GPT models)
  - Groq API key (for Llama models)

### Step 1: Clone or Download the Project

```bash
cd e:\Project Name
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or using uv (faster):

```bash
uv add requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## ⚙️ Configuration

The Streamlit interface provides real-time configuration options in the sidebar:

| Parameter            | Default                   | Description                            |
| -------------------- | ------------------------- | -------------------------------------- |
| **PDF Folder**       | `./data/pdf`              | Path to your PDF documents             |
| **Vector Store Dir** | `./data/vector_store`     | Chroma persistence directory           |
| **Embedding Model**  | `all-MiniLM-L6-v2`        | SentenceTransformers model             |
| **Collection Name**  | `pdf_documents`           | Chroma collection name                 |
| **Chunk Size**       | 1000                      | Tokens per chunk                       |
| **Chunk Overlap**    | 200                       | Overlap between chunks                 |
| **Top K**            | 3                         | Number of documents to retrieve        |
| **Min Similarity**   | 0.0                       | Minimum similarity threshold (0.0-1.0) |
| **Use Groq LLM**     | False                     | Toggle between Groq and OpenAI         |
| **Groq Model**       | `llama-3.3-70b-versatile` | Groq model selection                   |

---

## 🎮 Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Step-by-Step Usage

1. **Add Documents**

   - Place your PDF files in `data/pdf/` folder
   - Or modify the path in the sidebar

2. **Index Documents**

   - Click "Index / Re-index dataset" button
   - Wait for processing to complete
   - System will:
     - Load all documents from the folder
     - Split into chunks
     - Generate embeddings
     - Store in vector database

3. **Query the System**

   - Enter your question in the main text area
   - System will:
     - Retrieve relevant documents
     - Generate an answer using LLM
     - Display results with source documents

4. **Adjust Parameters**
   - Experiment with chunk sizes
   - Adjust Top K for more/fewer results
   - Modify similarity threshold
   - Switch between LLMs

---

## 📚 Usage Guide

### Basic Query Example

```
User: "What are the key features mentioned in the documents?"

System:
1. Retrieves top 3 most relevant documents
2. Sends query + context to LLM
3. Returns: Generated answer + source documents
```

### Advanced Features

#### Adjusting Similarity Threshold

- Lower threshold (0.0-0.3): More results, less relevant
- Higher threshold (0.7-1.0): Fewer results, highly relevant

#### Changing Embedding Models

- `all-MiniLM-L6-v2`: Lightweight, fast (default)
- `all-mpnet-base-v2`: Larger, more accurate
- `multi-qa-MiniLM-L6-cos-v1`: Optimized for Q&A

#### Using Different LLMs

- **OpenAI GPT-4**: Most capable, requires API key
- **Groq Llama-3.3-70b**: Fast, free tier available

---

## 🛠 Technologies Used

| Technology               | Purpose                             |
| ------------------------ | ----------------------------------- |
| **LangChain**            | LLM orchestration and RAG framework |
| **Streamlit**            | Web UI framework                    |
| **Chroma**               | Vector database                     |
| **SentenceTransformers** | Embedding generation                |
| **Groq API**             | LLM inference (alternative)         |
| **OpenAI API**           | LLM inference                       |
| **PyPDF/PyMuPDF**        | PDF document loading                |
| **LangGraph**            | Workflow orchestration              |

---

## 🔧 Troubleshooting

### Issue: "No documents found"

**Solution:**

- Ensure PDF files are in `data/pdf/` directory
- Check file format is supported (PDF, TXT, CSV, DOCX, XLS)
- Verify file permissions

### Issue: "Embedding model not found"

**Solution:**

```bash
pip install sentence-transformers --upgrade
```

### Issue: "API Key Error"

**Solution:**

- Verify `.env` file exists in project root
- Check API keys are valid
- Ensure no extra spaces in `.env` file

### Issue: "Slow performance"

**Solution:**

- Reduce chunk size
- Decrease top_k retrieval count
- Use lighter embedding model (`all-MiniLM-L6-v2`)

### Issue: "Out of memory"

**Solution:**

- Use CPU-based FAISS instead of GPU
- Reduce document corpus size
- Increase chunk size (process fewer chunks)

---

## 📝 Example Queries

1. "Summarize the main points from these documents"
2. "What is the conclusion of the document?"
3. "List all the benefits mentioned"
4. "How does the system work?"
5. "What are the technical specifications?"

---

## 🤝 Contributing

Feel free to extend this RAG system with:

- Additional document formats
- More LLM providers
- Advanced caching strategies
- Database optimizations
- Custom embedding models

---

## 📄 License

This project is provided as-is for educational and commercial use.

---

## 📧 Support

For issues or questions, please refer to the troubleshooting section or check the documentation of the respective libraries:

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Chroma Documentation](https://docs.trychroma.com/)

---

**Last Updated:** November 2025
