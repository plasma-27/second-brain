import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERSIST_DIR = "./second_brain_db"
COLLECTION_NAME = "second_brain_memories"


class SecondBrainDB:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        persist_dir: str = PERSIST_DIR,
    ):
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=embedding_api_key or api_key,
            base_url=(embedding_base_url or base_url) or None,
            check_embedding_ctx_length=False,  # Sends raw strings instead of tiktoken IDs
        )
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
        )

    def add_memory(
        self,
        content: str,
        title: str,
        summary: str,
        source_type: str,
        tags: List[str],
        url: Optional[str] = None,
    ) -> str:
        """Chunk (if long) and store a memory with structured metadata."""
        memory_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat(timespec="seconds")

        # Chroma metadata values must be primitives (str, int, float, bool)
        base_metadata = {
            "memory_id": memory_id,
            "title": title,
            "summary": summary,
            "source_type": source_type,  # "fact" | "note" | "bookmark"
            "tags": ", ".join(tags),
            "url": url or "",
            "created_at": timestamp,
        }

        # Keep short facts/notes intact; split long articles/bookmarks
        if len(content) > 1000:
            chunks = self.splitter.split_text(content)
            docs = [
                Document(
                    page_content=f"Title: {title}\nSummary: {summary}\n\n{chunk}",
                    metadata={**base_metadata, "chunk_index": i},
                )
                for i, chunk in enumerate(chunks)
            ]
        else:
            docs = [
                Document(
                    page_content=content,
                    metadata={**base_metadata, "chunk_index": 0},
                )
            ]

        self.vectorstore.add_documents(docs)
        return memory_id

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve top-k semantically similar chunks."""
        return self.vectorstore.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = 3):
        """Retrieve top-k chunks with L2 distance scores (lower = more similar)."""
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def delete_memory(self, memory_id: str) -> int:
        """Delete all chunks belonging to a specific memory_id."""
        data = self.vectorstore.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(data["ids"], data["metadatas"])
            if meta and meta.get("memory_id") == memory_id
        ]
        if ids_to_delete:
            self.vectorstore.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def list_memories(self) -> List[Dict[str, Any]]:
        """Return deduplicated list of stored memories sorted newest-first."""
        data = self.vectorstore.get(include=["metadatas", "documents"])
        seen: Dict[str, Dict[str, Any]] = {}

        for doc_text, meta in zip(data["documents"], data["metadatas"]):
            if not meta:
                continue
            mid = meta.get("memory_id", "unknown")
            if mid not in seen:
                seen[mid] = {
                    "memory_id": mid,
                    "title": meta.get("title", "Untitled"),
                    "summary": meta.get("summary", doc_text[:150]),
                    "source_type": meta.get("source_type", "note"),
                    "tags": meta.get("tags", ""),
                    "url": meta.get("url", ""),
                    "created_at": meta.get("created_at", ""),
                    "chunks": 1,
                }
            else:
                seen[mid]["chunks"] += 1

        return sorted(
            seen.values(),
            key=lambda x: x["created_at"],
            reverse=True,
        )