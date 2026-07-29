"""FastAPI surface for the platform (Chapter 2: control/data plane, delivery).

Run locally:

    uvicorn llm_platform.app:app --reload

Endpoints:
    GET  /healthz          -- liveness/readiness
    POST /v1/chat          -- synchronous request/response
    POST /v1/chat/stream   -- server-sent-events token streaming
    GET  /v1/models        -- model registry (control plane view)

FastAPI/uvicorn are optional. Importing this module without FastAPI installed
raises a clear error only when you actually try to build the app.
"""
from __future__ import annotations

import json
from typing import Optional

from .data import seed_store
from .pipeline import LLMPipeline
from .schemas import ChatRequest
from .config import MODEL_REGISTRY


def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - only when fastapi missing
        raise RuntimeError(
            "FastAPI is required for the HTTP app. Install with "
            "`pip install fastapi uvicorn`."
        ) from exc

    app = FastAPI(title="LLM Platform", version="0.1.0")
    pipeline = LLMPipeline(store=seed_store())

    class ChatBody(BaseModel):
        query: str
        user_id: str = "anonymous"
        use_rag: bool = True
        max_tokens: int = Field(256, ge=1, le=4096)
        temperature: float = Field(0.2, ge=0.0, le=2.0)
        force_model: Optional[str] = None

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "models": list(MODEL_REGISTRY)}

    @app.get("/v1/models")
    def models():
        return {k: vars(v) for k, v in MODEL_REGISTRY.items()}

    @app.post("/v1/chat")
    def chat(body: ChatBody):
        req = ChatRequest(**body.model_dump())
        return pipeline.handle(req).as_dict()

    @app.post("/v1/chat/stream")
    def chat_stream(body: ChatBody):
        req = ChatRequest(**body.model_dump(), stream=True)

        def gen():
            for chunk in pipeline.stream(req):
                yield f"data: {json.dumps({'delta': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


# Module-level app for `uvicorn llm_platform.app:app`.
try:  # pragma: no cover - convenience only
    app = create_app()
except RuntimeError:
    app = None
