# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# REST server mode for FactReasoner.
#
# Exposes the FactReasoner pipeline (all_pairs NLI mode, v2 graph shape) as a
# FastAPI HTTP service so it can be called from any HTTP client without having to
# fork a Python process per assessment.
#
# The backend and Merlin path are configured once at startup via CLI flags or
# environment variables; the per-request payload carries only the query/response
# to assess (or precomputed atoms + contexts for file-mode scoring).
#
# Requires the `server` optional-dependency group:
#   uv sync --extra server
#
# Run:
#   fact-reasoner-server \
#       --backend rits \
#       --merlin-path /path/to/merlin \
#       --host 0.0.0.0 \
#       --port 8080
#
# Or equivalently via environment variables:
#   FR_BACKEND=rits FR_MERLIN_PATH=/path/to/merlin \
#       fact-reasoner-server --host 0.0.0.0 --port 8080

from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "The 'server' optional dependencies are required to run fact-reasoner-server. "
        "Install them with:  uv sync --extra server"
    ) from _e

from fact_reasoner.assessor import FactReasoner
from fact_reasoner.backends import build_backend
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.nli import NLIExtractor
from fact_reasoner.core.nli_config import NLI_MODES, get_pair_config
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.core.retriever import ContextRetriever, SourceRetriever
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.summarizer import ContextSummarizer
from fact_reasoner.runner import _FR_VERSIONS

# ---------------------------------------------------------------------------
# Global server state (populated by the lifespan hook).
# ---------------------------------------------------------------------------

_server_state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pydantic request / response models.
# ---------------------------------------------------------------------------


class AssessRequest(BaseModel):
    """Assess a response from scratch (atomize → retrieve → score)."""

    query: str = Field(..., description="The input query.")
    response: str = Field(..., description="The LLM-generated response to assess.")
    topic: str | None = Field(None, description="Optional topic hint.")
    nli_mode: str = Field(
        "all_pairs",
        description=(
            "NLI candidate-pair preset. 'all_pairs' scores every enumerated pair "
            "(highest fidelity, reproduces published numbers). 'fast' applies the "
            "provenance preset (fewer LLM calls, same graph semantics)."
        ),
    )
    pipeline_version: str = Field(
        "v2",
        description="FactReasoner graph shape: 'v1', 'v2' (default), or 'v3'.",
    )


class AssessFromDictRequest(BaseModel):
    """Score precomputed atoms + contexts without re-running atomization or retrieval.

    The payload format is the same as the dict produced by
    ``FactReasoner.to_json()`` / ``pipeline_to_json()``, i.e. it must contain
    the keys ``input``, ``output``, ``atoms``, and ``contexts``.
    """

    data: dict[str, Any] = Field(
        ...,
        description=(
            "Precomputed pipeline dict as returned by FactReasoner.to_json(). "
            "Must contain 'input', 'output', 'atoms', and 'contexts' keys."
        ),
    )
    nli_mode: str = Field(
        "all_pairs",
        description="NLI candidate-pair preset ('all_pairs' or 'fast').",
    )
    pipeline_version: str = Field(
        "v2",
        description="FactReasoner graph shape: 'v1', 'v2' (default), or 'v3'.",
    )


# ---------------------------------------------------------------------------
# Shared pipeline-component factory.
# ---------------------------------------------------------------------------


def _build_pipeline(
    nli_mode: str,
    pipeline_version: str,
    state: dict[str, Any],
    with_retriever: bool,
) -> tuple[FactReasoner, dict]:
    """Construct a fresh FactReasoner and the matching build kwargs.

    A new pipeline object is created per request so that concurrent requests do
    not share mutable state. The backend and heavy NLI/atomizer components are
    re-used from ``state`` (they are stateless).

    Args:
        nli_mode: NLI preset name.
        pipeline_version: Graph-shape version key.
        state: The server-global state dict (holds backend, merlin_path, etc.).
        with_retriever: When True, wire up context retrieval (live mode). When
            False, skip the retriever (file / precomputed mode).

    Returns:
        A (pipeline, build_kwargs) pair ready to pass to ``pipeline.build()``.
    """
    if nli_mode not in NLI_MODES:
        raise ValueError(
            f"Unknown nli_mode {nli_mode!r}. Expected one of {list(NLI_MODES)}."
        )
    if pipeline_version not in _FR_VERSIONS:
        raise ValueError(
            f"Unknown pipeline_version {pipeline_version!r}. "
            f"Expected one of {list(_FR_VERSIONS)}."
        )

    rel_atom_ctx, rel_ctx_ctx, remove_dups, ctx_per_atom = _FR_VERSIONS[
        pipeline_version
    ]

    context_retriever = None
    if with_retriever:
        query_builder = QueryBuilder(state["backend"]) if state["use_query_builder"] else None
        source_retriever = SourceRetriever(
            service_type=state["service_type"],
            top_k=state["top_k"],
            cache_dir=state["cache_dir"],
            fetch_text=True,
            query_builder=query_builder,
            num_workers=state["num_workers"],
        )
        context_retriever = ContextRetriever(
            retriever=source_retriever,
            context_summarizer=state["context_summarizer"],
            num_workers=state["num_workers"],
        )

    pipeline = FactReasoner(
        atom_extractor=state["atom_extractor"],
        atom_reviser=state["atom_reviser"],
        nli_extractor=state["nli_extractor"],
        context_retriever=context_retriever,
        context_summarizer=state["context_summarizer"],
        merlin_path=state["merlin_path"],
        nli_pair_config=get_pair_config(nli_mode),
        nli_cache_dir=state.get("nli_cache_dir"),
    )

    build_kwargs = {
        "remove_duplicates": remove_dups,
        "contexts_per_atom_only": ctx_per_atom,
        "rel_atom_context": rel_atom_ctx,
        "rel_context_context": rel_ctx_ctx,
        "use_fast_retriever": True,
        "summarize_contexts": state["use_summarizer"],
    }
    return pipeline, build_kwargs


# ---------------------------------------------------------------------------
# FastAPI application.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialize shared components once on startup; tear down on shutdown."""
    cfg = _server_state["config"]

    backend = build_backend(
        cfg.backend,
        model_id=cfg.model_id or None,
        base_url=cfg.base_url or None,
    )

    _server_state.update(
        backend=backend,
        merlin_path=cfg.merlin_path,
        service_type=cfg.service_type,
        top_k=cfg.top_k,
        num_workers=cfg.num_workers,
        cache_dir=cfg.cache_dir or None,
        use_summarizer=cfg.use_summarizer,
        use_query_builder=cfg.use_query_builder,
        nli_cache_dir=cfg.nli_cache_dir or None,
        atom_extractor=Atomizer(backend),
        atom_reviser=Reviser(backend),
        nli_extractor=NLIExtractor(backend),
        context_summarizer=ContextSummarizer(backend),
    )

    print(
        f"[FactReasoner server] Ready — backend={cfg.backend}, "
        f"merlin={cfg.merlin_path}",
        flush=True,
    )
    yield
    print("[FactReasoner server] Shutting down.", flush=True)


app = FastAPI(
    title="FactReasoner",
    description=(
        "REST API for the FactReasoner factuality assessment pipeline. "
        "POST a query/response to /assess to score it from scratch, or POST "
        "precomputed atoms+contexts to /assess/from_dict to skip retrieval."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    """Return ``{"status": "ok"}`` when the server is ready."""
    return {"status": "ok"}


@app.post("/assess", summary="Assess a response from scratch")
async def assess(req: AssessRequest) -> JSONResponse:
    """Atomize ``response``, retrieve contexts, and score factuality.

    This mirrors the live path in
    ``docs/examples/assessors/ex_factreasoner_all_pairs.py``: the response is
    atomized, atoms are revised (decontextualized), contexts are retrieved from
    the configured search service, and the FactReasoner graph is scored with
    Merlin.

    The ``nli_mode`` field selects the NLI candidate-pair preset:
    - ``"all_pairs"`` (default) — highest fidelity, reproduces published numbers.
    - ``"fast"`` — provenance preset, far fewer LLM calls for the same graph.
    """
    try:
        pipeline, build_kwargs = _build_pipeline(
            req.nli_mode, req.pipeline_version, _server_state, with_retriever=True
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        await pipeline.build(
            query=req.query,
            response=req.response,
            topic=req.topic,
            has_atoms=False,
            has_contexts=False,
            revise_atoms=True,
            **build_kwargs,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline build failed: {exc}") from exc

    results, marginals = pipeline.score()
    pipeline_dict = pipeline.to_json()
    return JSONResponse(content={"results": results, "marginals": marginals, "pipeline": pipeline_dict})


@app.post("/assess/from_dict", summary="Score precomputed atoms and contexts")
async def assess_from_dict(req: AssessFromDictRequest) -> JSONResponse:
    """Score a response whose atoms and contexts are already computed.

    The ``data`` field must be a dict in the same format as
    ``FactReasoner.to_json()`` (keys: ``input``, ``output``, ``atoms``,
    ``contexts``). No retrieval or atomization is performed — only NLI relation
    extraction and Merlin inference.

    This mirrors the ``--input-file`` path in
    ``docs/examples/assessors/ex_factreasoner_all_pairs.py``.
    """
    try:
        pipeline, build_kwargs = _build_pipeline(
            req.nli_mode, req.pipeline_version, _server_state, with_retriever=False
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        pipeline.from_dict_with_contexts(req.data)
    except (KeyError, AssertionError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid pipeline dict: {exc}",
        ) from exc

    try:
        await pipeline.build(
            has_atoms=True,
            has_contexts=True,
            revise_atoms=False,
            **build_kwargs,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline build failed: {exc}") from exc

    results, marginals = pipeline.score()
    return JSONResponse(content={"results": results, "marginals": marginals})


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fact-reasoner-server",
        description=(
            "Run FactReasoner as a REST API server (requires the 'server' extra). "
            "Backend and Merlin path can also be set via environment variables: "
            "FR_BACKEND, FR_MODEL_ID, FR_BASE_URL, FR_MERLIN_PATH, FR_SERVICE_TYPE, "
            "FR_CACHE_DIR, FR_NLI_CACHE_DIR."
        ),
    )

    # Server
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (dev).")

    # Backend — mirrors the CLI flags in cli.py.
    parser.add_argument(
        "--backend",
        default=os.getenv("FR_BACKEND", "ollama"),
        choices=["ollama", "rits", "vllm", "openai"],
        help="Backend to use (default: ollama, or FR_BACKEND env).",
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("FR_MODEL_ID"),
        help="Model id (or FR_MODEL_ID env).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("FR_BASE_URL"),
        help="API endpoint (or FR_BASE_URL env).",
    )
    parser.add_argument(
        "--merlin-path",
        default=os.getenv("FR_MERLIN_PATH"),
        required=not os.getenv("FR_MERLIN_PATH"),
        help="Path to the Merlin binary (or FR_MERLIN_PATH env).",
    )

    # Pipeline
    parser.add_argument(
        "--service-type",
        default=os.getenv("FR_SERVICE_TYPE", "google"),
        choices=["google", "wikipedia", "chromadb"],
        help="Retrieval service (default: google).",
    )
    parser.add_argument("--cache-dir", default=os.getenv("FR_CACHE_DIR"), help="Retriever cache dir.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k contexts per atom.")
    parser.add_argument("--num-workers", type=int, default=4, help="Context retrieval parallelism.")
    parser.add_argument("--use-summarizer", action="store_true", help="Summarize contexts.")
    parser.add_argument("--use-query-builder", action="store_true", help="Use QueryBuilder.")
    parser.add_argument("--nli-cache-dir", default=os.getenv("FR_NLI_CACHE_DIR"), help="NLI verdict cache dir.")

    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Stash config in global state so the lifespan hook can read it.
    _server_state["config"] = args

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
