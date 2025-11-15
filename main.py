# app.py
import os
import uuid
from pathlib import Path
from typing import List, Any, Dict

import streamlit as st
import numpy as np

# Document loaders and splitters (same imports as you used)
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vector DB
from sentence_transformers import SentenceTransformer
import chromadb

# Optional Groq LLM (keeps same pattern; optional)
from dotenv import load_dotenv
load_dotenv()
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

# -------------------------
# Helper: load all supported documents from ./data/pdf (and other supported files)
# -------------------------
def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain documents.
    Supported: PDF, TXT, CSV, Excel, DOCX, JSON
    """
    data_path = Path(data_dir).resolve()
    st.write(f"[DEBUG] Data path: {data_path}")
    documents = []

    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    st.write(f"[DEBUG] Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        st.write(f"[DEBUG] Loading PDF: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            st.write(f"[DEBUG] Loaded {len(loaded)} pages from {pdf_file.name}")
            documents.extend(loaded)
        except Exception as e:
            st.error(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # TXT files
    txt_files = list(data_path.glob('**/*.txt'))
    st.write(f"[DEBUG] Found {len(txt_files)} TXT files")
    for txt_file in txt_files:
        st.write(f"[DEBUG] Loading TXT: {txt_file.name}")
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source_file'] = txt_file.name
                doc.metadata['file_type'] = 'txt'
            st.write(f"[DEBUG] Loaded {len(loaded)} parts from {txt_file.name}")
            documents.extend(loaded)
        except Exception as e:
            st.error(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    st.write(f"[DEBUG] Found {len(csv_files)} CSV files")
    for csv_file in csv_files:
        st.write(f"[DEBUG] Loading CSV: {csv_file.name}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source_file'] = csv_file.name
                doc.metadata['file_type'] = 'csv'
            st.write(f"[DEBUG] Loaded {len(loaded)} docs from {csv_file.name}")
            documents.extend(loaded)
        except Exception as e:
            st.error(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # Excel files
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    st.write(f"[DEBUG] Found {len(xlsx_files)} Excel files")
    for xlsx_file in xlsx_files:
        st.write(f"[DEBUG] Loading Excel: {xlsx_file.name}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source_file'] = xlsx_file.name
                doc.metadata['file_type'] = 'xlsx'
            st.write(f"[DEBUG] Loaded {len(loaded)} docs from {xlsx_file.name}")
            documents.extend(loaded)
        except Exception as e:
            st.error(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    # DOCX files
    docx_files = list(data_path.glob('**/*.docx'))
    st.write(f"[DEBUG] Found {len(docx_files)} Word files")
    for docx_file in docx_files:
        st.write(f"[DEBUG] Loading DOCX: {docx_file.name}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source_file'] = docx_file.name
                doc.metadata['file_type'] = 'docx'
            st.write(f"[DEBUG] Loaded {len(loaded)} docs from {docx_file.name}")
            documents.extend(loaded)
        except Exception as e:
            st.error(f"[ERROR] Failed to load DOCX {docx_file}: {e}")

    # JSON files
    json_files = list(data_path.glob('**/*.json'))
    st.write(f"[DEBUG] Found {len(json_files)} JSON files")
    for json_file in json_files:
        st.write(f"[DEBUG] Loading JSON: {json_file.name}")
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source_file'] = json_file.name
                doc.metadata['file_type'] = 'json'
            st.write(f"[DEBUG] Loaded {len(loaded)} docs from {json_file.name}")
            documents.extend(loaded)
        except Exception as e:
            st.error(f"[ERROR] Failed to load JSON {json_file}: {e}")

    st.write(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# -------------------------
# Text splitter
# -------------------------
def split_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    st.write(f"[DEBUG] Split {len(documents)} docs into {len(split_docs)} chunks")
    return split_docs

# -------------------------
# Embedding Manager
# -------------------------
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            st.write(f"[DEBUG] Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            st.write(f"[DEBUG] Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            st.error(f"[ERROR] Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        st.write(f"[DEBUG] Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings)
        st.write(f"[DEBUG] Generated embeddings shape: {embeddings.shape}")
        return embeddings

# -------------------------
# Vector Store (ChromaDB)
# -------------------------
class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "./data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            # Use chromadb.PersistentClient with path
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            st.write(f"[DEBUG] Vector store initialized at: {self.persist_directory}, collection: {self.collection_name}")
        except Exception as e:
            st.error(f"[ERROR] Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        st.write(f"[DEBUG] Adding {len(documents)} documents to vector store...")
        ids, metadatas, docs_texts, embeddings_list = [], [], [], []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            docs_texts.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=docs_texts
            )
            st.success(f"[DEBUG] Successfully added {len(documents)} documents to vector store.")
        except Exception as e:
            st.error(f"[ERROR] Error adding documents to vector store: {e}")
            raise

# -------------------------
# Retriever
# -------------------------
class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        st.write(f"[DEBUG] Retrieving for query: '{query}' (top_k={top_k})")
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            retrieved_docs = []
            if results.get('documents') and results['documents'][0]:
                docs = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, docs, metadatas, distances)):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                st.write(f"[DEBUG] Retrieved {len(retrieved_docs)} documents after thresholding")
            else:
                st.write("[DEBUG] No documents found in vector store for this query.")
            return retrieved_docs
        except Exception as e:
            st.error(f"[ERROR] Error during retrieval: {e}")
            return []

# -------------------------
# Optional Groq LLM wrapper
# -------------------------
class GroqLLM:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not HAS_GROQ:
            st.warning("[WARN] langchain_groq or langchain_core not available; LLM disabled.")
            self.llm = None
            return
        if not self.api_key:
            st.warning("[WARN] GROQ_API_KEY not set; LLM calls may fail.")
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )
        st.write(f"[DEBUG] Initialized Groq LLM: {self.model_name}")

    def generate_response(self, query: str, context: str) -> str:
        if not HAS_GROQ or not self.llm:
            st.warning("[WARN] LLM not available. Returning context preview.")
            return context[:2000] + ("..." if len(context) > 2000 else "")
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
        )
        formatted = prompt_template.format(context=context, question=query)
        messages = [HumanMessage(content=formatted)]
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            st.error(f"[ERROR] Groq LLM invocation error: {e}")
            return context[:2000]

# -------------------------
# Reflect heuristic
# -------------------------
def reflect_answer(query: str, answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_score = max([d['similarity_score'] for d in retrieved_docs], default=0.0)
    contains_query_tokens = any(tok.lower() in answer.lower() for tok in query.split()[:6])
    relevance = "HIGH" if (max_score >= 0.3 and contains_query_tokens) else ("MEDIUM" if max_score >= 0.15 else "LOW")
    return {"max_similarity": float(max_score), "contains_query_tokens_in_answer": contains_query_tokens, "relevance": relevance}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="RAG Agent", layout="wide")
st.title("RAG Pipeline — Ingest → Index → Retrieve → Answer → Reflect")

with st.sidebar:
    st.header("Configuration")
    # Default paths match your repo structure
    data_pdf_dir = st.text_input("PDF folder (data/pdf)", value="./data/pdf")
    persist_dir = st.text_input("Vector store dir (data/vector_store)", value="./data/vector_store")
    embedding_model = st.text_input("Embedding model", value="all-MiniLM-L6-v2")
    collection_name = st.text_input("Chroma collection name", value="pdf_documents")
    chunk_size = st.number_input("Chunk size", value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", value=200, step=25)
    top_k = st.slider("Top K", 1, 10, 3)
    min_score = st.slider("Min similarity threshold", 0.0, 1.0, 0.0, 0.05)
    reindex_btn = st.button("Index / Re-index dataset")
    use_groq = st.checkbox("Use Groq LLM (optional)", value=False)
    groq_model_name = st.text_input("Groq model name", value="llama-3.3-70b-versatile")

# Cached resource factories (names and keywords are explicit and consistent)
@st.cache_resource(show_spinner=False)
def get_embedding_manager(model_name: str):
    return EmbeddingManager(model_name=model_name)

@st.cache_resource(show_spinner=False)
def get_vectorstore(collection_name: str, persist_directory: str):
    return VectorStore(collection_name=collection_name, persist_directory=persist_directory)

# Instantiate cached resources (use explicit keyword args)
embedding_manager = get_embedding_manager(model_name=embedding_model)
vectorstore = get_vectorstore(collection_name=collection_name, persist_directory=persist_dir)
retriever = RAGRetriever(vectorstore, embedding_manager)
groq_llm = GroqLLM(model_name=groq_model_name, api_key=os.getenv("GROQ_API_KEY")) if use_groq else None

# Re-index flow: ingest -> split -> embed -> persist
if reindex_btn:
    st.info("Starting data ingestion and indexing...")
    docs = load_all_documents(data_pdf_dir)
    if not docs:
        st.warning("No documents found under the specified data directory.")
    else:
        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = [d.page_content for d in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vectorstore.add_documents(chunks, embeddings)
        st.success("Indexing complete!")

# Query UI
st.subheader("Ask a question")
query = st.text_input("Enter question here", "")
if st.button("Run RAG") and query.strip():
    st.markdown("#### PLAN node")
    plan = {"query": query, "need_retrieval": True}
    st.json(plan)

    st.markdown("#### RETRIEVE node")
    retrieved = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not retrieved:
        st.warning("No context retrieved from vector store. Make sure you indexed documents and the persist dir is correct.")
    else:
        for i, doc in enumerate(retrieved):
            st.write(f"Result #{i+1} — score: {doc['similarity_score']:.4f} — source: {doc['metadata'].get('source_file','unknown')}")
            st.write(doc['content'][:800] + ("..." if len(doc['content']) > 800 else ""))
            st.write(doc['metadata'])

    st.markdown("#### ANSWER node")
    context = "\n\n".join([d['content'] for d in retrieved]) if retrieved else ""
    if not context:
        answer_text = "No relevant context found to answer the question."
        st.write(answer_text)
    else:
        if groq_llm:
            st.write("Generating answer using Groq LLM...")
            answer_text = groq_llm.generate_response(query, context)
        else:
            st.write("Groq not enabled. Returning context preview as answer.")
            answer_text = context[:2000] + ("..." if len(context) > 2000 else "")
        st.text_area("Answer", value=answer_text, height=250)

    st.markdown("#### REFLECT node")
    reflection = reflect_answer(query, answer_text, retrieved)
    st.json(reflection)

    st.markdown("#### Sources & Confidence")
    sources = [{
        "source": d['metadata'].get('source_file', d['metadata'].get('source', 'unknown')),
        "page": d['metadata'].get('page', 'unknown'),
        "score": d['similarity_score'],
        "preview": d['content'][:200] + "..."
    } for d in retrieved]
    st.write(sources)
    st.success(f"Agent finished. Relevance: {reflection['relevance']} (max sim: {reflection['max_similarity']:.3f})")

st.markdown("---")
st.markdown("**Tips:** Make sure your PDF files live in `./data/pdf/`. After adding or updating files, click **Index / Re-index dataset**. Then ask a question and click **Run RAG**.")
