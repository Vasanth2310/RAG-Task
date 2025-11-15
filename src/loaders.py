from pathlib import Path
from typing import List, Any
import os

# Optional: use Streamlit writes if running under Streamlit UI
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def _write(msg: str):
    if _HAS_ST:
        st.write(msg)
    else:
        # silent fallback
        pass

def _error(msg: str):
    if _HAS_ST:
        st.error(msg)
    else:
        # fallback to print for CLI visibility
        print("ERROR:", msg)

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files (pdf, txt, csv, xlsx, docx, json) from data_dir.
    Returns a list of LangChain Document objects.
    """
    data_path = Path(data_dir).resolve()
    _write(f"[DEBUG] Data path: {data_path}")
    documents = []

    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    _write(f"[DEBUG] Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        _write(f"[DEBUG] Loading PDF: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            _write(f"[DEBUG] Loaded {len(loaded)} pages from {pdf_file.name}")
            documents.extend(loaded)
        except Exception as e:
            _error(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # TXT files
    txt_files = list(data_path.glob('**/*.txt'))
    _write(f"[DEBUG] Found {len(txt_files)} TXT files")
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source_file'] = txt_file.name
                doc.metadata['file_type'] = 'txt'
            documents.extend(loaded)
        except Exception as e:
            _error(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    _write(f"[DEBUG] Found {len(csv_files)} CSV files")
    for csv_file in csv_files:
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source_file'] = csv_file.name
                doc.metadata['file_type'] = 'csv'
            documents.extend(loaded)
        except Exception as e:
            _error(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # Excel files
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    _write(f"[DEBUG] Found {len(xlsx_files)} Excel files")
    for xlsx_file in xlsx_files:
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source_file'] = xlsx_file.name
                doc.metadata['file_type'] = 'xlsx'
            documents.extend(loaded)
        except Exception as e:
            _error(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    # DOCX files
    docx_files = list(data_path.glob('**/*.docx'))
    _write(f"[DEBUG] Found {len(docx_files)} Word files")
    for docx_file in docx_files:
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source_file'] = docx_file.name
                doc.metadata['file_type'] = 'docx'
            documents.extend(loaded)
        except Exception as e:
            _error(f"[ERROR] Failed to load DOCX {docx_file}: {e}")

    # JSON files
    json_files = list(data_path.glob('**/*.json'))
    _write(f"[DEBUG] Found {len(json_files)} JSON files")
    for json_file in json_files:
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source_file'] = json_file.name
                doc.metadata['file_type'] = 'json'
            documents.extend(loaded)
        except Exception as e:
            _error(f"[ERROR] Failed to load JSON {json_file}: {e}")

    _write(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents
