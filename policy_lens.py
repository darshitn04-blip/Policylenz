"""Core retrieval and rules logic for the PolicyLens student project."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None

    @property
    def citation(self) -> str:
        return f"{self.source}{f', p. {self.page}' if self.page else ''}"


def extract_pdf_chunks(file_path: str | Path, chunk_size: int = 850, overlap: int = 130,
                       source_name: str | None = None) -> list[Chunk]:
    """Parse a PDF page-by-page, then make overlapping chunks for retrieval."""
    reader = PdfReader(str(file_path))
    chunks: list[Chunk] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        for piece in chunk_text(text, chunk_size, overlap):
            chunks.append(Chunk(piece, source_name or Path(file_path).name, page_no))
    return chunks


def chunk_text(text: str, chunk_size: int = 850, overlap: int = 130) -> list[str]:
    """Lightweight chunker designed for untidy document text."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    result, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        result.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return result


def load_demo_chunks(data_dir: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in Path(data_dir).glob("*.md"):
        for piece in chunk_text(path.read_text(encoding="utf-8")):
            chunks.append(Chunk(piece, path.stem.replace("_", " ").title()))
    return chunks


class Retriever:
    """Local semantic retrieval with an offline TF-IDF fallback for easy setup."""
    def __init__(self, chunks: Iterable[Chunk]):
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("Add at least one readable document before asking a question.")
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            vectors = self.model.encode([c.text for c in self.chunks], normalize_embeddings=True)
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(np.asarray(vectors, dtype="float32"))
            self.backend = "SentenceTransformer + FAISS"
        except Exception:
            # Keeps the project easy to run on machines without PyTorch/FAISS.
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform(c.text for c in self.chunks)
            self.backend = "TF-IDF fallback (install requirements for semantic retrieval)"

    def search(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]:
        if hasattr(self, "index"):
            vector = self.model.encode([question], normalize_embeddings=True)
            scores, ids = self.index.search(np.asarray(vector, dtype="float32"), min(k, len(self.chunks)))
            return [(self.chunks[i], float(score)) for i, score in zip(ids[0], scores[0]) if i != -1]
        scores = (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()
        ids = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in ids]


def answer_question(question: str, evidence: list[tuple[Chunk, float]]) -> str:
    """Answer only from retrieved evidence. Uses OpenAI when configured, otherwise extractive RAG."""
    context = "\n\n".join(f"[{i + 1}] {chunk.text}" for i, (chunk, _) in enumerate(evidence))
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        prompt = f"""You explain Indian government schemes in clear plain English.
Use ONLY the evidence below. If it does not answer the question, say so. Do not invent benefits or eligibility.
Give a short structured answer and cite claims using [1], [2], etc.

Question: {question}
Evidence:\n{context}"""
        return OpenAI().chat.completions.create(
            model=os.getenv("POLICYLENS_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}], temperature=0.1,
        ).choices[0].message.content or "No answer generated."
    # Honest, inspectable fallback for classroom demos without an API key.
    snippets = "\n".join(f"- {chunk.text} [{i + 1}]" for i, (chunk, _) in enumerate(evidence))
    return "**Relevant information from the uploaded documents**\n\n" + snippets


def check_eligibility(scheme: str, profile: dict) -> dict:
    """Deterministic decision table. Rules are intentionally visible and easy to extend."""
    scheme = scheme.lower()
    income = float(profile.get("annual_income") or 0)
    land = float(profile.get("land_hectares") or 0)
    age = int(profile.get("age") or 0)
    documents = set(profile.get("documents", []))
    if "kisan" in scheme:
        missing = {"Aadhaar", "Bank account", "Land record"} - documents
        eligible = land > 0 and not missing
        reason = "You reported cultivable land and the required identity, bank and land records." if eligible else "PM-KISAN needs a recorded landholding plus Aadhaar, bank account and land record details."
    elif "ayushman" in scheme:
        missing = {"Aadhaar"} - documents
        eligible = age >= 0 and not missing
        reason = "Your details pass this simplified checker; final inclusion depends on the official beneficiary database." if eligible else "An Aadhaar document is needed for this simplified checker."
    else:
        eligible, missing = False, set()
        reason = "No rules are configured for this document yet. Review the cited guidelines or add a decision-table rule."
    return {"eligible": eligible, "reason": reason, "missing_documents": sorted(missing), "income_recorded": income}
