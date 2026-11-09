"""FastAPI adapter that routes requests through the single answer pipeline."""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from trustrag.offline.embed_index import OfflineFAISSIndex
from trustrag.offline.knowledge_store import KnowledgeStore
from trustrag.pipeline import TrustRAGPipeline


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    system: str = "S2"
    llm: str = "ollama"
    as_of: str | None = None
    live_check: bool = False


class AskResponse(BaseModel):
    answer: str
    confidence: float
    faithfulness: float | None
    verification_status: str
    category: str | None = None
    volatility: str | None = None
    evidence_strategy: str | None = None
    freshness: dict[str, Any]
    agreement: str
    conflicts: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    unsupported_claims: list[dict[str, Any]]
    flags: list[str]
    latency_ms: float
    trace: dict[str, Any]
    verification_required: bool = False


def create_app(config_path: str | Path = "trustrag/config.yaml") -> FastAPI:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    app = FastAPI(title="TrustRAG Travel Advisory", version="0.1.0")
    app.state.config_path = config_path
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> dict[str, Any]:
        return TrustRAGPipeline(app.state.config_path).answer(
            request.query,
            system=request.system,
            llm=request.llm,
            as_of=request.as_of,
            live_check=request.live_check,
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        paths = config["paths"]
        index_paths = [
            Path(paths.get("primary_faiss_path", "data/store/primary.faiss")),
            Path(paths.get("secondary_faiss_path", "data/store/secondary.faiss")),
        ]
        docs = 0
        try:
            with KnowledgeStore.from_config(config_path) as store:
                docs = store.connection.execute("SELECT COUNT(*) FROM documents WHERE status='active'").fetchone()[0]
        except FileNotFoundError:
            pass
        return {
            "status": "ok",
            "index_loaded": all(path.exists() for path in index_paths),
            "documents": docs,
            "embedding_model": config.get("embedding", {}).get("model_name"),
            "default_llm": config.get("llm", {}).get("default", "mock"),
        }

    @app.post("/admin/reindex")
    def reindex(x_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
        expected = os.environ.get("ADMIN_TOKEN")
        if not expected or not x_admin_token or not hmac.compare_digest(expected, x_admin_token):
            raise HTTPException(status_code=401, detail="Invalid admin token")
        indexer = OfflineFAISSIndex.from_config(config_path)
        return indexer.build_from_directory(config.get("paths", {}).get("corpus_dir", "data/corpus"))

    return app


app = create_app()