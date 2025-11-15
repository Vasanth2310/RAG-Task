import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from src.loaders import load_all_documents
from src.splitter import split_documents
from src.embeddings import EmbeddingManager
from src.vectorstore import VectorStore
from src.retriever import RAGRetriever, reflect_answer
from src.llm import GroqLLM

st.set_page_config(page_title="RAG Agent", layout="wide")
st.title("RAG Pipeline")

with st.sidebar:
    st.header("Configuration")
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

@st.cache_resource(show_spinner=False)
def get_embedding_manager(model_name: str):
    return EmbeddingManager(model_name=model_name)

@st.cache_resource(show_spinner=False)
def get_vectorstore(collection_name: str, persist_directory: str):
    return VectorStore(collection_name=collection_name, persist_directory=persist_directory)

embedding_manager = get_embedding_manager(model_name=embedding_model)
vectorstore = get_vectorstore(collection_name=collection_name, persist_directory=persist_dir)
retriever = RAGRetriever(vectorstore, embedding_manager)
groq_llm = GroqLLM(model_name=groq_model_name, api_key=os.getenv("GROQ_API_KEY")) if use_groq else None

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
