"""RAG over the seed knowledge base with LangChain (Chapters 13-16).

Builds a LangChain ``InMemoryVectorStore`` from the seed documents, retrieves the
top-k, assembles a grounded prompt, and answers with citations via an LCEL chain.
Uses OllamaEmbeddings(nomic-embed-text) when the Ollama backend is active,
otherwise deterministic hash embeddings.
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

from .config import get_chat_model, get_embeddings
from .seed import SEED_DOCS
from .telemetry import Telemetry, estimate_tokens, timed

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the provided context. If it is insufficient, say so. "
               "Cite the [doc_id] of each source you use."),
    ("human", "# Context\n{context}\n\n# Question\n{question}\n\n# Answer"),
])


def build_vector_store(*, backend: Optional[str] = None) -> InMemoryVectorStore:
    docs = [Document(page_content=d["text"],
                     metadata={"doc_id": d["doc_id"], "source": d["source"]})
            for d in SEED_DOCS]
    return InMemoryVectorStore.from_documents(docs, get_embeddings(backend=backend))


class RAGPipeline:
    def __init__(self, *, backend: Optional[str] = None, k: int = 4) -> None:
        self.backend = backend
        self.k = k
        self.store = build_vector_store(backend=backend)
        self.model = get_chat_model(temperature=0.1, backend=backend)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        return self.store.similarity_search(query, k=k or self.k)

    def answer(self, query: str, k: Optional[int] = None) -> dict:
        tel = Telemetry(tier="rag")
        with timed(tel):
            docs = self.retrieve(query, k=k)
            context = "\n\n".join(f"[{d.metadata['doc_id']}] {d.page_content}" for d in docs)
            chain = _PROMPT | self.model | StrOutputParser()
            answer = chain.invoke({"context": context, "question": query})
        tel.input_tokens = estimate_tokens(context + query)
        tel.output_tokens = estimate_tokens(answer)
        tel.extra["retrieved_docs"] = len(docs)
        return {
            "query": query,
            "answer": answer,
            "citations": [d.metadata["doc_id"] for d in docs],
            "contexts": [{"doc_id": d.metadata["doc_id"], "source": d.metadata["source"],
                          "text": d.page_content} for d in docs],
            "telemetry": tel.as_dict(),
        }
