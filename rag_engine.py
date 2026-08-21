"""
Module for curriculum document indexing and vector-similarity context retrieval.
Utilizes Google Gemini embedding models and local JSON persistence for RAG operations.
"""

import os
import json
import glob
from typing import List, Optional, Dict, Any
import numpy as np
import pypdf
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.join(BASE_DIR, "study_materials")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store.json")

os.makedirs(STUDY_DIR, exist_ok=True)

def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    """
    Computes cosine similarity between two numerical embedding vectors.

    Args:
        vector_a (List[float]): First numerical embedding.
        vector_b (List[float]): Second numerical embedding.

    Returns:
        float: Cosine similarity coefficient in range [-1.0, 1.0].
    """
    arr_a, arr_b = np.array(vector_a), np.array(vector_b)
    dot_product = np.dot(arr_a, arr_b)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

class RAGEngine:
    """
    Manages document parsing, recursive text splitting, embedding generation,
    and semantic context retrieval for the adaptive tutoring pipeline.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.documents: List[Dict[str, Any]] = []
        self._load_store()

    def _load_store(self) -> None:
        if os.path.exists(VECTOR_STORE_PATH):
            try:
                with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as file_handle:
                    self.documents = json.load(file_handle)
            except Exception:
                self.documents = []

    def _save_store(self) -> None:
        with open(VECTOR_STORE_PATH, "w", encoding="utf-8") as file_handle:
            json.dump(self.documents, file_handle, ensure_ascii=False, indent=2)

    def _get_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Generates dense vector embeddings using the configured Google Gemini model.

        Args:
            text (str): Content to embed.
            task_type (str): Retrieval task type classification.

        Returns:
            List[float]: High-dimensional embedding vector.
        """
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type=task_type
        )
        return result["embedding"]

    def extract_text_from_file(self, filepath: str) -> str:
        """
        Extracts raw text content from Markdown, Plain Text, or PDF files.

        Args:
            filepath (str): Absolute or relative path to the target document.

        Returns:
            str: Extracted textual contents.
        """
        extension = os.path.splitext(filepath)[1].lower()
        if extension in [".txt", ".md"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as file_handle:
                return file_handle.read()
        elif extension == ".pdf":
            extracted_text = ""
            reader = pypdf.PdfReader(filepath)
            for page in reader.pages:
                text_content = page.extract_text()
                if text_content:
                    extracted_text += text_content + "\n"
            return extracted_text
        return ""

    def ingest_study_materials(self) -> int:
        """
        Parses all curriculum documents in study_materials and indexes new chunks.

        Returns:
            int: Total count of active indexed document chunks.
        """
        supported_files = []
        for ext in ["*.pdf", "*.txt", "*.md"]:
            supported_files.extend(glob.glob(os.path.join(STUDY_DIR, ext)))

        existing_ids = {doc["id"] for doc in self.documents}
        new_chunks = 0

        for filepath in supported_files:
            filename = os.path.basename(filepath)
            content = self.extract_text_from_file(filepath)
            if not content.strip():
                continue

            chunks = self.text_splitter.split_text(content)
            for idx, chunk in enumerate(chunks):
                doc_id = f"{filename}_chunk_{idx}"
                if doc_id in existing_ids:
                    continue

                emb = self._get_embedding(chunk, task_type="retrieval_document")
                self.documents.append({
                    "id": doc_id,
                    "source": filename,
                    "chunk_index": idx,
                    "text": chunk,
                    "embedding": emb
                })
                existing_ids.add(doc_id)
                new_chunks += 1

        if new_chunks > 0:
            self._save_store()

        return len(self.documents)

    def retrieve_relevant_context(
        self, query: str, top_k: int = 3, min_similarity: float = 0.5
    ) -> Optional[str]:
        """
        Retrieves top relevant curriculum text passages based on cosine similarity scoring.

        Args:
            query (str): Input query text or user prompt.
            top_k (int): Maximum number of top passages to retrieve.
            min_similarity (float): Threshold similarity score required for inclusion.

        Returns:
            Optional[str]: Formatted context block with source references, or None if no match.
        """
        if not self.documents:
            self.ingest_study_materials()

        if not self.documents:
            return None

        query_emb = self._get_embedding(query, task_type="retrieval_query")

        scored_docs = []
        for doc in self.documents:
            sim = cosine_similarity(query_emb, doc["embedding"])
            if sim >= min_similarity:
                scored_docs.append((sim, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_docs[:top_k]

        if not top_results:
            return None

        context_blocks = []
        for score, doc in top_results:
            context_blocks.append(
                f"[Source: {doc['source']} (Relevance: {int(score * 100)}%)]\n{doc['text']}"
            )

        return "\n\n---\n\n".join(context_blocks)
