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

"""Job store for the FactReasoner REST server.

Defines a ``JobStore`` Protocol and a single SQLite-backed implementation:

- ``SQLiteJobStore`` — file-based, suitable for single-node / dev deployments.

Use ``make_job_store(backend, url)`` to construct an instance.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Dict, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class JobStore(Protocol):
    """Minimal key-value interface required by the fact-check job lifecycle."""

    def __setitem__(self, job_id: str, value: Dict[str, Any]) -> None: ...

    def get(self, job_id: str, default: Any = None) -> Any: ...

    def __contains__(self, job_id: str) -> bool: ...


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------


class SQLiteJobStore:
    """Thread-safe, multi-process job store backed by SQLite (WAL mode).

    Multiple Gunicorn worker processes all open the same database file, so
    every worker can read jobs created by any other worker.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._ensure_table()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection (created lazily)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _ensure_table(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id  TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Public dict-like interface
    # ------------------------------------------------------------------

    def __setitem__(self, job_id: str, value: Dict[str, Any]) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO jobs (job_id, payload) VALUES (?, ?)",
            (job_id, json.dumps(value)),
        )
        conn.commit()

    def get(self, job_id: str, default: Any = None) -> Any:
        row = self._conn().execute(
            "SELECT payload FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return json.loads(row[0]) if row else default

    def __contains__(self, job_id: str) -> bool:
        return self.get(job_id) is not None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_job_store(backend: str, url: str) -> JobStore:
    """Construct a :class:`SQLiteJobStore` for the given *url*.

    Args:
        backend: Storage backend identifier.  Only ``"sqlite"`` is supported.
        url: File path for the SQLite database (e.g. ``./jobs.db``).
    """
    if backend != "sqlite":
        raise ValueError(f"Unsupported job store backend {backend!r}.")
    return SQLiteJobStore(url)
