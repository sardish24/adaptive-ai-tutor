import os
import json
import glob
from typing import List, Optional, Dict, Any
import numpy as np
import pypdf
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.join(BASE_DIR, "study_materials")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store.json")

os.makedirs(STUDY_DIR, exist_ok=True)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

class RAGEngine:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.documents: List[Dict[str, Any]] = []
        self._load_store()

    def _load_store(self):
        if os.path.exists(VECTOR_STORE_PATH):
            try:
                with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception:
                self.documents = []

    def _save_store(self):
        with open(VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def _get_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generates embedding using Google Gemini embedding model."""
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type=task_type
        )
        return result['embedding']

    def extract_text_from_file(self, filepath: str) -> str:
        """Extracts plain text from text, markdown, or PDF files."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext in [".txt", ".md"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            text = ""
            reader = pypdf.PdfReader(filepath)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        return ""

    def ingest_study_materials(self) -> int:
        """Indexes all study files from ./study_materials into persistent vector store."""
        supported_files = []
        for ext in ["*.pdf", "*.txt", "*.md"]:
            supported_files.extend(glob.glob(os.path.join(STUDY_DIR, ext)))

        existing_ids = {doc['id'] for doc in self.documents}
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

    def retrieve_relevant_context(self, query: str, top_k: int = 3, min_similarity: float = 0.5) -> Optional[str]:
        """Calculates cosine similarity and returns top-k matching curriculum excerpts."""
        if not self.documents:
            self.ingest_study_materials()

        if not self.documents:
            return None

        query_emb = self._get_embedding(query, task_type="retrieval_query")

        scored_docs = []
        for doc in self.documents:
            sim = cosine_similarity(query_emb, doc['embedding'])
            if sim >= min_similarity:
                scored_docs.append((sim, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_docs[:top_k]

        if not top_results:
            return None

        context_blocks = []
        for score, doc in top_results:
            context_blocks.append(f"[Source: {doc['source']} (Relevance: {int(score*100)}%)]\n{doc['text']}")

        return "\n\n---\n\n".join(context_blocks)
