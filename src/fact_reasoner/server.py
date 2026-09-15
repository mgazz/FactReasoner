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

# Legacy REST server for FactReasoner with document-management and async job pattern.
#
# Endpoints preserved from the original server_old.py:
#   POST  /fact-check/start         — async job: atomize → retrieve → score
#   GET   /fact-check/status/{id}   — poll job result
#   GET   /display_graph/           — render fact graph as interactive HTML (gravis)
#   GET   /graph/json/{graph_id}    — serve saved graph JSON in multiple formats
#   GET   /                         — root health message
#
# Removed:
#   POST /correct/                   — depended on lfqa_corrector (not present)
#   POST /documents/upload           — skipped (docling/chromadb management)
#   DELETE /documents/delete/{fn}   — skipped (docling/chromadb management)
#   DELETE /vector-db/delete/{fn}   — skipped (docling/chromadb management)
#
# Requires the `server` optional-dependency group:
#   uv sync --extra server
#
# Run:
#   fact-reasoner-server-old \
#       --backend rits \
#       --merlin-path /path/to/merlin \
#       --doc-db-path ./chroma_db_docs \
#       --doc-db-collection my_collection \
#       --host 0.0.0.0 \
#       --port 8001

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, Optional

import networkx as nx

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "The 'server' optional dependencies are required to run fact-reasoner-server-old. "
        "Install them with:  uv sync --extra server"
    ) from _e

from fact_reasoner.job_store import JobStore, make_job_store
from fact_reasoner.assessor import FactReasoner
from fact_reasoner.backends import build_backend
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.nli import NLIExtractor
from fact_reasoner.core.nli_config import NLI_MODES, get_pair_config
from fact_reasoner.core.retriever import ContextRetriever, SourceRetriever
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.summarizer import ContextSummarizer
from fact_reasoner.runner import _FR_VERSIONS

# ---------------------------------------------------------------------------
# Global server state (populated by the lifespan hook).
# ---------------------------------------------------------------------------

_server_state: dict[str, Any] = {}


def _gravis_d3(*args, **kwargs):
    """Lazy wrapper — defers the gravis import to the first /display_graph/ call."""
    try:
        from gravis import d3
    except ImportError as _e:
        raise ImportError(
            "gravis is required for the /display_graph/ endpoint. "
            "Install it with:  uv sync --extra server"
        ) from _e
    return d3(*args, **kwargs)

# Initialised in _lifespan.
job_store: JobStore  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pydantic request / response models.
# ---------------------------------------------------------------------------


class FactCheckRequest(BaseModel):
    question: str
    answer: str


class JobCreationResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GraphFormat(str, Enum):
    full = "full"
    scored = "scored"
    metadata = "metadata"
    annotation = "annotation"


# ---------------------------------------------------------------------------
# Pipeline factory (same pattern as server.py).
# ---------------------------------------------------------------------------


def _build_pipeline(
    nli_mode: str,
    pipeline_version: str,
    state: dict[str, Any],
) -> tuple[FactReasoner, dict]:
    """Construct a fresh FactReasoner wired to the configured SourceRetriever.

    A new pipeline object is created per request so concurrent requests do not
    share mutable state.  The backend and heavy components are re-used from
    ``state`` (they are stateless).

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

    source_retriever = SourceRetriever(
        service_type=state["source_retriever"],
        collection_name=state["doc_db_collection"],
        persist_dir=state["doc_db_path"],
        top_k=state["top_k"],
    )
    context_retriever = ContextRetriever(
        retriever=source_retriever,
        context_summarizer=state["context_summarizer"],
    )

    pipeline = FactReasoner(
        atom_extractor=state["atom_extractor"],
        atom_reviser=state["atom_reviser"],
        nli_extractor=state["nli_extractor"],
        context_retriever=context_retriever,
        context_summarizer=state["context_summarizer"],
        merlin_path=state["merlin_path"],
        nli_pair_config=get_pair_config(nli_mode),
    )

    build_kwargs = {
        "remove_duplicates": remove_dups,
        "contexts_per_atom_only": ctx_per_atom,
        "rel_atom_context": rel_atom_ctx,
        "rel_context_context": rel_ctx_ctx,
        "use_fast_retriever": True,
        "summarize_contexts": True,
    }
    return pipeline, build_kwargs


# ---------------------------------------------------------------------------
# Process-pool worker (runs in a separate OS process).
# ---------------------------------------------------------------------------


def _hydrate_worker_state(state_snapshot: dict) -> dict:
    """Reconstruct live objects (backend, extractors) inside the worker process.

    ``state_snapshot`` contains only picklable scalars; the heavy objects must
    be built fresh because they cannot cross the process boundary.
    """
    backend = build_backend(
        state_snapshot["backend_type"],
        model_id=state_snapshot["model_id"],
        base_url=state_snapshot["base_url"],
    )
    return {
        **state_snapshot,
        "backend": backend,
        "atom_extractor": Atomizer(backend),
        "atom_reviser": Reviser(backend),
        "nli_extractor": NLIExtractor(backend),
        "context_summarizer": ContextSummarizer(backend),
    }


def _fact_check_worker(
    job_id: str,
    question: str,
    answer: str,
    nli_mode: str,
    pipeline_version: str,
    state_snapshot: dict,
) -> dict:
    """Synchronous entry point executed in a worker process.

    Spins up a fresh event loop so that the async ``pipeline.build`` can run,
    then returns a result dict that the main process stores in *job_store*.
    """
    import asyncio as _asyncio
    import traceback as _traceback

    server_state = _hydrate_worker_state(state_snapshot)

    async def _run() -> dict:
        pipeline, build_kwargs = _build_pipeline(nli_mode, pipeline_version, server_state)

        await pipeline.build(
            query=question,
            response=answer,
            has_atoms=False,
            has_contexts=False,
            revise_atoms=True,
            **build_kwargs,
        )

        results, marginals = pipeline.score()

        graph_id = str(uuid.uuid4())
        graph_dir = server_state["graph_dir"]
        os.makedirs(graph_dir, exist_ok=True)

        graph_path = os.path.join(graph_dir, f"{graph_id}_graph.json")
        pipeline_path = os.path.join(graph_dir, f"{graph_id}_pipeline.json")
        annotation_path = os.path.join(graph_dir, f"{graph_id}_annotation.json")

        # --- scored graph (as_json uses source/target keys) ---
        graph_data = pipeline.fact_graph.as_json()

        # Stamp atom probabilities + labels from score results onto graph nodes.
        predictions = results.get("predictions", {})
        fscore_per_atom = results.get("factuality_score_per_atom", [])
        atom_scores: dict[str, float] = {
            atom_id: entry["score"]
            for item in fscore_per_atom
            for atom_id, entry in item.items()
        }
        for node in graph_data.get("nodes", []):
            if node.get("type") == "atom":
                nid = node["id"]
                if nid in atom_scores:
                    node["probability"] = atom_scores[nid]
                if nid in predictions:
                    node["label"] = predictions[nid]

        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=4)

        # --- full pipeline state ---
        pipeline_data = pipeline.to_json(json_file_path=pipeline_path)

        # --- annotation file (enriched graph for human review) ---
        atom_details = {a["id"]: a for a in pipeline_data.get("atoms", [])}
        context_details = {c["id"]: c for c in pipeline_data.get("contexts", [])}

        # Build atom → relevant context ids from graph edges (source=context, target=atom)
        relevant_contexts_for_atoms: dict[str, list[str]] = {}
        for edge in graph_data.get("edges", []):
            if edge.get("relation") in ("entailment", "contradiction"):
                context_id = edge.get("source")
                atom_id = edge.get("target")
                if atom_id and context_id:
                    relevant_contexts_for_atoms.setdefault(atom_id, []).append(context_id)

        annotated_nodes = []
        for node in graph_data.get("nodes", []):
            node = dict(node)  # shallow copy so we don't mutate graph_data
            nid = node.get("id")
            if node.get("type") == "atom" and nid in atom_details:
                node["text"] = atom_details[nid].get("text", "")
                node["label"] = predictions.get(nid, "N/A")
                node["relevant_contexts"] = [
                    context_details[cid]
                    for cid in relevant_contexts_for_atoms.get(nid, [])
                    if cid in context_details
                ]
                node["annotation"] = ""
                node["comment"] = ""
            elif node.get("type") == "context" and nid in context_details:
                ctx = context_details[nid]
                node["text"] = ctx.get("text", "")
                node["synthetic_summary"] = ctx.get("synthetic_summary", "")
            annotated_nodes.append(node)

        annotated_edges = []
        for edge in graph_data.get("edges", []):
            edge = dict(edge)  # shallow copy
            src = edge.get("source")
            tgt = edge.get("target")
            if src in context_details:
                edge["context_text"] = context_details[src].get("text", "")
                edge["context_summary"] = context_details[src].get("synthetic_summary", "")
            if tgt in atom_details:
                edge["atom_text"] = atom_details[tgt].get("text", "")
            edge["annotation"] = ""
            edge["comment"] = ""
            annotated_edges.append(edge)

        annotation_graph = {
            "question": question,
            "answer": answer,
            "nodes": annotated_nodes,
            "edges": annotated_edges,
        }
        with open(annotation_path, "w", encoding="utf-8") as f:
            json.dump(annotation_graph, f, indent=4)

        # --- derive supported/not-supported atom lists for the job result ---
        supported_atoms = []
        not_supported_atoms = []
        for atom_id, atom in pipeline.atoms.items():
            label = predictions.get(atom_id)
            entry = {"id": atom_id, "text": atom.text}
            if label == "S":
                supported_atoms.append(entry)
            else:
                not_supported_atoms.append(entry)

        return {
            "status": "completed",
            "result": {
                "fact_reasoner_score": results.get("factuality_score"),
                "supported_atoms": supported_atoms,
                "not_supported_atoms": not_supported_atoms,
                "contexts": pipeline_data.get("contexts", []),
                "graph_id": graph_id,
            },
        }

    try:
        return _asyncio.run(_run())
    except Exception:
        logging.exception("Worker process job %s failed.", job_id)
        return {
            "status": "failed",
            "error": _traceback.format_exc(),
        }


async def _dispatch_fact_check(
    job_id: str,
    question: str,
    answer: str,
    nli_mode: str,
    pipeline_version: str,
) -> None:
    """Submit the job to the process pool and update *job_store* when done."""
    loop = asyncio.get_running_loop()
    # Pass only picklable plain-dict state — no live objects.
    state_snapshot = {
        k: _server_state[k]
        for k in (
            "merlin_path",
            "graph_dir",
            "source_retriever",
            "doc_db_path",
            "doc_db_collection",
            "top_k",
        )
    }
    # Backend and heavy components must be re-created inside the worker because
    # they hold open HTTP connections / locks that can't cross process boundaries.
    cfg = _server_state["config"]
    state_snapshot["backend_type"] = cfg.backend
    state_snapshot["model_id"] = cfg.model_id or None
    state_snapshot["base_url"] = cfg.base_url or None

    try:
        result = await loop.run_in_executor(
            _server_state["process_pool"],
            _fact_check_worker,
            job_id,
            question,
            answer,
            nli_mode,
            pipeline_version,
            state_snapshot,
        )
        job_store[job_id] = result
    except Exception:
        import traceback
        logging.exception("Process-pool job %s failed.", job_id)
        job_store[job_id] = {
            "status": "failed",
            "error": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Graph helper.
# ---------------------------------------------------------------------------


def _load_graph_files(
    graph_id: str,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load the scored-graph JSON and full pipeline JSON for *graph_id*.

    Returns:
        ``(graph_data, pipeline_data)`` where ``pipeline_data`` may be ``None``
        if the pipeline file is absent.

    Raises:
        HTTPException 404 if the graph file is missing.
    """
    graph_dir = _server_state["graph_dir"]
    graph_path = os.path.join(graph_dir, f"{graph_id}_graph.json")
    pipeline_path = os.path.join(graph_dir, f"{graph_id}_pipeline.json")

    if not os.path.exists(graph_path):
        raise HTTPException(
            status_code=404, detail=f"Graph file not found for ID: {graph_id}"
        )

    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    pipeline_data = None
    if os.path.exists(pipeline_path):
        with open(pipeline_path, "r", encoding="utf-8") as f:
            pipeline_data = json.load(f)

    return graph_data, pipeline_data


# ---------------------------------------------------------------------------
# Lifespan hook.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialise shared components once on startup; tear down on shutdown."""
    global job_store

    cfg = _server_state["config"]

    # Initialise the job store (backend + URL driven by CLI / env vars).
    job_store = make_job_store(cfg.job_store_backend, cfg.job_db)

    backend = build_backend(
        cfg.backend,
        model_id=cfg.model_id or None,
        base_url=cfg.base_url or None,
    )

    workers = int(os.getenv("FR_POOL_WORKERS", "4"))
    process_pool = ProcessPoolExecutor(max_workers=workers)

    _server_state.update(
        backend=backend,
        merlin_path=cfg.merlin_path,
        graph_dir=cfg.graph_dir,
        source_retriever=cfg.source_retriever,
        doc_db_path=cfg.doc_db_path,
        doc_db_collection=cfg.doc_db_collection,
        top_k=cfg.top_k,
        atom_extractor=Atomizer(backend),
        atom_reviser=Reviser(backend),
        nli_extractor=NLIExtractor(backend),
        context_summarizer=ContextSummarizer(backend),
        process_pool=process_pool,
    )

    print(
        f"[FactReasoner server-old] Ready — backend={cfg.backend}, "
        f"merlin={cfg.merlin_path}, graph_dir={cfg.graph_dir}, "
        f"pool_workers={workers}",
        flush=True,
    )
    yield
    process_pool.shutdown(wait=False, cancel_futures=True)
    print("[FactReasoner server-old] Shutting down.", flush=True)


# ---------------------------------------------------------------------------
# FastAPI application.
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fact-Reasoning Service (legacy)",
    description=(
        "Legacy REST API for FactReasoner with async fact-check jobs and graph serving. "
        "POST a question/answer to /fact-check/start, then poll /fact-check/status/{job_id}."
    ),
    version="0.2.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Fact-check endpoints.
# ---------------------------------------------------------------------------


@app.post(
    "/fact-check/start",
    response_model=JobCreationResponse,
    summary="Start a new fact-checking job",
)
async def start_fact_check_job(request: FactCheckRequest) -> JobCreationResponse:
    """Queue a fact-checking job in the process pool and return its ``job_id`` immediately."""
    cfg = _server_state["config"]
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "pending"}
    asyncio.create_task(
        _dispatch_fact_check(
            job_id,
            request.question,
            request.answer,
            cfg.nli_mode,
            cfg.pipeline_version,
        )
    )
    return JobCreationResponse(job_id=job_id)


@app.get(
    "/fact-check/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Get the status and result of a job",
)
async def get_fact_check_status(job_id: str) -> JobStatusResponse:
    """Return the current status (``pending`` / ``completed`` / ``failed``) of a job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JobStatusResponse(job_id=job_id, **job)


# ---------------------------------------------------------------------------
# Graph endpoints.
# ---------------------------------------------------------------------------


@app.get(
    "/display_graph/",
    response_class=HTMLResponse,
    summary="Display a saved fact graph as interactive HTML",
)
async def display_graph(graph_id: str = Query(...)) -> HTMLResponse:
    """Render the fact graph for *graph_id* as an interactive gravis/d3 page."""
    graph_data, pipeline_data = _load_graph_files(graph_id)
    if pipeline_data is None:
        raise HTTPException(
            status_code=404,
            detail="Pipeline file required for HTML display not found.",
        )

    # Build an nx.DiGraph directly from the as_json() output (source/target keys).
    nx_graph = nx.DiGraph()
    for node in graph_data.get("nodes", []):
        nx_graph.add_node(node["id"], type=node.get("type"), probability=node.get("probability", 1.0))
    for edge in graph_data.get("edges", []):
        nx_graph.add_edge(
            edge["source"],
            edge["target"],
            label=edge.get("relation", ""),
            probability=edge.get("probability", 1.0),
        )

    # Atom scores already stamped on graph nodes during background job.
    atom_scores = {
        node["id"]: node.get("probability", None)
        for node in graph_data.get("nodes", [])
        if node.get("type") == "atom"
    }

    # Build per-node hover text from pipeline atoms/contexts.
    node_texts: dict[str, str] = {}
    for atom in pipeline_data.get("atoms", []):
        aid = atom["id"]
        score = atom_scores.get(aid, "N/A")
        formatted = f"{score:.4f}" if isinstance(score, float) else str(score)
        node_texts[aid] = f"Atom Score: {formatted}\n{atom.get('text', '')}"

    for ctx in pipeline_data.get("contexts", []):
        cid = ctx["id"]
        node_texts[cid] = (
            f"Title: {ctx.get('title', 'N/A')}\n"
            f"Text: {ctx.get('synthetic_summary', ctx.get('text', ''))}"
        )

    for node_id in nx_graph.nodes():
        if node_id in node_texts:
            nx_graph.nodes[node_id]["hover"] = node_texts[node_id]

    fig = _gravis_d3(
        nx_graph,
        show_edge_label=True,
        edge_label_data_source="label",
        edge_curvature=0.2,
        use_edge_size_normalization=False,
        edge_size_factor=1.5,
        use_node_size_normalization=True,
        node_size_factor=2.5,
    )
    return HTMLResponse(content=fig.to_html())


@app.get(
    "/graph/json/{graph_id}",
    response_class=JSONResponse,
    summary="Get graph data as JSON in various formats",
)
async def get_graph_json(
    graph_id: str,
    format: GraphFormat = GraphFormat.full,
    blind_annotation: bool = False,
) -> JSONResponse:
    """Return graph data for *graph_id* in one of four formats:

    - ``scored``     — raw scored-graph nodes + edges (with probabilities/labels).
    - ``annotation`` — enriched annotation graph (from the annotation file).
    - ``metadata``   — full pipeline JSON from ``pipeline.to_json()``.
    - ``full``       — scored graph nodes enriched with atom/context text from the pipeline.
    """
    graph_data, pipeline_data = _load_graph_files(graph_id)

    if format == GraphFormat.scored:
        return JSONResponse(content=graph_data)

    if pipeline_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline file not found for ID: {graph_id}, cannot return '{format.value}' format.",
        )

    if format == GraphFormat.annotation:
        annotation_path = os.path.join(
            _server_state["graph_dir"], f"{graph_id}_annotation.json"
        )
        if not os.path.exists(annotation_path):
            raise HTTPException(
                status_code=404,
                detail=f"Annotation file not found for ID: {graph_id}",
            )
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation_data = json.load(f)

        if blind_annotation:
            blind_data = copy.deepcopy(annotation_data)
            for node in blind_data.get("nodes", []):
                node.pop("probability", None)
                if node.get("type") == "atom":
                    node.pop("label", None)
                for ctx in node.get("relevant_contexts", []):
                    ctx.pop("probability", None)
            for edge in blind_data.get("edges", []):
                edge.pop("relation", None)
                edge.pop("probability", None)
            return JSONResponse(content=blind_data)

        return JSONResponse(content=annotation_data)

    if format == GraphFormat.metadata:
        return JSONResponse(content=pipeline_data)

    # format == GraphFormat.full: enrich graph nodes with text from pipeline.
    atom_details = {a["id"]: a for a in pipeline_data.get("atoms", [])}
    context_details = {c["id"]: c for c in pipeline_data.get("contexts", [])}

    enriched_nodes = []
    for node in graph_data.get("nodes", []):
        node = dict(node)
        nid = node.get("id")
        if node.get("type") == "atom" and nid in atom_details:
            node["text"] = atom_details[nid].get("text", "")
        elif node.get("type") == "context" and nid in context_details:
            ctx_info = context_details[nid]
            node["title"] = ctx_info.get("title", "N/A")
            node["summary"] = ctx_info.get("synthetic_summary", "")
            node["link"] = ctx_info.get("link", "")
        enriched_nodes.append(node)

    return JSONResponse(
        content={
            "graph": {
                "nodes": enriched_nodes,
                "edges": graph_data.get("edges", []),
            },
            "metadata": pipeline_data,
        }
    )


# ---------------------------------------------------------------------------
# Root endpoint.
# ---------------------------------------------------------------------------


@app.get("/", summary="Root endpoint")
async def root() -> dict[str, str]:
    return {
        "message": (
            "Fact-Checking Service (legacy) is running. "
            "Visit /docs for API documentation."
        )
    }


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fact-reasoner-server-old",
        description=(
            "Run the legacy FactReasoner REST API server (requires the 'server' extra). "
            "Backend and paths can also be set via environment variables: "
            "FR_BACKEND, FR_MODEL_ID, FR_BASE_URL, FR_MERLIN_PATH, "
            "FR_SOURCE_RETRIEVER, FR_GRAPH_DIR, FR_DOC_DB_PATH, FR_DOC_DB_COLLECTION."
        ),
    )

    # Server
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001).")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("FR_WORKERS", "1")),
        help="Number of Gunicorn worker processes (default: 1 or FR_WORKERS env).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (dev only, forces single worker).",
    )

    # Backend
    parser.add_argument(
        "--backend",
        default=os.getenv("FR_BACKEND", "ollama"),
        choices=["ollama", "rits", "vllm", "openai"],
        help="Inference backend (default: ollama or FR_BACKEND env).",
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

    # Job store
    parser.add_argument(
        "--job-store-backend",
        default="sqlite",
        choices=["sqlite"],
        help="Job store backend (default: sqlite).",
    )
    parser.add_argument(
        "--job-db",
        default=os.getenv("FR_JOB_DB", "./jobs.db"),
        help="SQLite database file path (default: ./jobs.db or FR_JOB_DB env).",
    )

    # Storage
    parser.add_argument(
        "--source-retriever",
        default=os.getenv("FR_SOURCE_RETRIEVER", "chromadb"),
        choices=["chromadb", "google", "wikipedia"],
        help="Source retrieval backend (default: chromadb or FR_SOURCE_RETRIEVER env).",
    )
    parser.add_argument(
        "--graph-dir",
        default=os.getenv("FR_GRAPH_DIR", "./graphs"),
        help="Directory to save graph JSON files (default: ./graphs or FR_GRAPH_DIR env).",
    )
    parser.add_argument(
        "--doc-db-path",
        default=os.getenv("FR_DOC_DB_PATH", "./chroma_db_docs"),
        help="Path to the ChromaDB persist directory for fact-check retrieval "
             "(default: ./chroma_db_docs or FR_DOC_DB_PATH env).",
    )
    parser.add_argument(
        "--doc-db-collection",
        default=os.getenv("FR_DOC_DB_COLLECTION", "akd_document_collection"),
        help="ChromaDB collection name (default: akd_document_collection or FR_DOC_DB_COLLECTION env).",
    )

    # Pipeline defaults (can be overridden per-request via the FactCheckRequest body in future)
    parser.add_argument("--top-k", type=int, default=3, help="Top-k contexts per atom (default: 3).")
    parser.add_argument(
        "--nli-mode",
        default="all_pairs",
        choices=list(NLI_MODES),
        help="NLI candidate-pair preset (default: all_pairs).",
    )
    parser.add_argument(
        "--pipeline-version",
        default="v2",
        choices=list(_FR_VERSIONS),
        help="FactReasoner graph-shape version (default: v2).",
    )

    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    _server_state["config"] = args

    if args.reload or args.workers == 1:
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    else:
        from gunicorn.app.base import BaseApplication

        class _StandaloneApp(BaseApplication):
            def __init__(self, application, options):
                self.application = application
                self.options = options
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key, value)

            def load(self):
                return self.application

        _StandaloneApp(
            app,
            {
                "bind": f"{args.host}:{args.port}",
                "workers": args.workers,
                "worker_class": "uvicorn.workers.UvicornWorker",
                "timeout": int(os.getenv("FR_WORKER_TIMEOUT", "600")),
                "accesslog": "-",
                "errorlog": "-",
            },
        ).run()


if __name__ == "__main__":
    main()
