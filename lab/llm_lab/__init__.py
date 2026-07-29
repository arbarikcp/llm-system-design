"""llm_lab -- a LangChain-powered, Streamlit-testable lab for the
"LLM System Design" ebook.

Every system from the book is implemented here with LangChain (LCEL runnables,
prompt templates, retrievers, tools) and is testable interactively in the
Streamlit UI (``app.py``).

Backends (see ``config.py``):
  * Ollama   -- ChatOllama(llama3.1) + OllamaEmbeddings(nomic-embed-text) on your machine.
  * Offline  -- a deterministic OfflineChatModel + HashEmbeddings, so the whole lab
                runs and is testable with no server, no GPU, and no API key.

Set ``LAB_BACKEND=ollama`` (or toggle it in the UI) to use real local models.
"""
from .config import get_chat_model, get_embeddings, backend_status

__all__ = ["get_chat_model", "get_embeddings", "backend_status"]
__version__ = "0.1.0"
