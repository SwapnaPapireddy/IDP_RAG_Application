import json, os, re, time, uuid
from pathlib import Path
import boto3
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

load_dotenv()

class DocumentRAG:
    """S3 + Textract ingestion; local SentenceTransformers + FAISS + Hugging Face RAG."""
    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket = os.getenv("S3_BUCKET", "").strip()
        if not self.bucket or self.bucket == "your-existing-bucket-name":
            raise ValueError("Set S3_BUCKET in .env to an existing S3 bucket.")
        self.data_dir = Path(os.getenv("INDEX_DIR", "data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path, self.index_path = self.data_dir / "chunks.json", self.data_dir / "faiss.index"
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
        self.overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
        self.top_k = int(os.getenv("TOP_K", "4"))
        self.s3 = boto3.client("s3", region_name=self.region)
        self.textract = boto3.client("textract", region_name=self.region)
        self.embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        self.generator = hf_pipeline("text2text-generation", model=os.getenv("HF_LLM_MODEL", "google/flan-t5-base"))
        try:
            import faiss
        except ImportError as e:
            raise RuntimeError("FAISS missing. Run pip install -r requirements.txt") from e
        self.faiss = faiss
        self.chunks, self.index = [], None
        self._load()

    def _load(self):
        if self.meta_path.exists():
            self.chunks = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.index_path.exists():
            self.index = self.faiss.read_index(str(self.index_path))
        elif self.chunks:
            v = np.asarray(self.embedder.encode([x["text"] for x in self.chunks], normalize_embeddings=True), dtype="float32")
            self.index = self.faiss.IndexFlatIP(v.shape[1]); self.index.add(v)

    def _save(self):
        self.meta_path.write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.index is not None:
            self.faiss.write_index(self.index, str(self.index_path))

    @staticmethod
    def _pages(blocks):
        pages = {}
        for b in blocks:
            if b.get("BlockType") == "LINE":
                pages.setdefault(int(b.get("Page", 1)), []).append(b.get("Text", ""))
        return [{"page": p, "text": "\n".join(lines)} for p, lines in sorted(pages.items())]

    def _pdf_ocr(self, key):
        job = self.textract.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": self.bucket, "Name": key}}
        )["JobId"]
        deadline = time.time() + int(os.getenv("TEXTRACT_TIMEOUT_SECONDS", "900"))
        while time.time() < deadline:
            result = self.textract.get_document_text_detection(JobId=job)
            status = result.get("JobStatus")
            if status == "FAILED":
                raise RuntimeError("Textract failed to process this PDF.")
            if status == "SUCCEEDED":
                blocks, token = result.get("Blocks", []), result.get("NextToken")
                while token:
                    more = self.textract.get_document_text_detection(JobId=job, NextToken=token)
                    blocks.extend(more.get("Blocks", [])); token = more.get("NextToken")
                return self._pages(blocks)
            time.sleep(3)
        raise TimeoutError("Textract timed out. Try a smaller PDF.")

    def _chunks(self, text):
        text = re.sub(r"[ \t]+", " ", text).strip()
        out, start = [], 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                split = text.rfind(" ", start, end)
                if split > start + self.chunk_size // 2: end = split
            part = text[start:end].strip()
            if part: out.append(part)
            if end >= len(text): break
            start = max(end - self.overlap, start + 1)
        return out

    def ingest(self, filename, content):
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
        key = f"uploads/{uuid.uuid4().hex[:10]}_{name}"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=content)
        if Path(name).suffix.lower() == ".pdf":
            pages = self._pdf_ocr(key)
        else:
            result = self.textract.detect_document_text(Document={"S3Object": {"Bucket": self.bucket, "Name": key}})
            pages = self._pages(result.get("Blocks", []))
        added = []
        for page in pages:
            for part in self._chunks(page["text"]):
                added.append({"document": name, "s3_key": key, "page": page["page"], "text": part})
        if not added: raise RuntimeError("Textract returned no text for this document.")
        vectors = np.asarray(self.embedder.encode([x["text"] for x in added], normalize_embeddings=True), dtype="float32")
        if self.index is None: self.index = self.faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors); self.chunks.extend(added); self._save()
        return {"chunks": len(added), "s3_key": key}

    def answer(self, question):
        if self.index is None or not self.chunks: raise ValueError("Upload and index a document first.")
        q = np.asarray(self.embedder.encode([question], normalize_embeddings=True), dtype="float32")
        scores, ids = self.index.search(q, min(self.top_k, len(self.chunks)))
        selected = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                c = self.chunks[int(idx)].copy(); c["score"] = float(score); selected.append(c)
        context = "\n\n".join(f"[{c['document']}, page {c['page']}]\n{c['text']}" for c in selected)
        prompt = ("Answer only from the context. If it does not contain the answer, say so. Do not invent facts.\n\n"
                  f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
        generated = self.generator(prompt, max_new_tokens=220, do_sample=False, truncation=True)[0]["generated_text"].strip()
        return generated, selected

    def list_documents(self):
        return sorted({c["document"] for c in self.chunks})

    def clear_index(self):
        self.chunks, self.index = [], None
        for p in (self.meta_path, self.index_path):
            if p.exists(): p.unlink()
