"""Local FastAPI sidecar for cited precedent RAG.

Schedule XER files stay in the browser. The UI only posts narrative + filters +
blocker titles — never the XER bytes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# load ProjectLens/.env before we touch Gemini / LangSmith
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from .graph import run_precedent_query  # noqa: E402


class PrecedentQuery(BaseModel):
    problem: str = Field(..., min_length=8, description="Live change problem / narrative snippet")
    narrative: str | None = None
    sector: str | None = None
    phase: str | None = None
    type: str | None = Field(default=None, description="Change type, e.g. Scope change")
    blockers: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=10)
    summarise: bool = True


app = FastAPI(
    title="ProjectLens precedent RAG",
    description="Cited precedent retrieval for change assurance. Not a decision authority.",
    version="0.1.0",
)

# browser product is often served from 127.0.0.1:8000 — allow that origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "https://matthewpaver.github.io",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "gemini": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        "langsmith": bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")),
        "project": os.getenv("LANGSMITH_PROJECT", "projectlens-precedent-rag"),
    }


@app.post("/precedents/query")
def query_precedents(body: PrecedentQuery) -> dict[str, Any]:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY missing in ProjectLens/.env — retrieval needs Gemini embeddings.",
        )
    payload = body.model_dump()
    try:
        return run_precedent_query(
            payload,
            limit=body.limit,
            summarise=body.summarise,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    host = os.getenv("PRECEDENT_RAG_HOST", "127.0.0.1")
    port = int(os.getenv("PRECEDENT_RAG_PORT", "8787"))
    uvicorn.run(
        "Processing.precedent_rag.server:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
