"""SQLite-backed checkpoint and domain-trace persistence seams."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol, Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from contractgraph.application_models import AnswerResult, RunLimits, TraceEvent


class TraceRepository(Protocol):
    def create_run(self, run_id: str, question: str, limits: RunLimits) -> None: ...

    def append_events(self, run_id: str, events: Sequence[TraceEvent]) -> None: ...

    def complete_run(self, result: AnswerResult) -> None: ...

    def read_run(self, run_id: str) -> dict[str, object]: ...


class CheckpointStore(Protocol):
    saver: BaseCheckpointSaver

    def close(self) -> None: ...


class SQLiteCheckpointStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self.saver = SqliteSaver(self._connection)
        self.saver.setup()

    def close(self) -> None:
        self._connection.close()


class SQLiteTraceRepository:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._setup()

    def _setup(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
                run_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                answer TEXT,
                iterations INTEGER NOT NULL,
                model_calls INTEGER NOT NULL,
                limits_json TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS trace_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES agent_runs(run_id),
                sequence INTEGER NOT NULL,
                node TEXT NOT NULL,
                event_type TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                details_json TEXT NOT NULL,
                UNIQUE(run_id, sequence)
            );
            """
        )
        columns = {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(agent_runs)")
        }
        if "result_json" not in columns:
            self._connection.execute(
                "ALTER TABLE agent_runs ADD COLUMN result_json TEXT"
            )
        self._connection.commit()

    def create_run(self, run_id: str, question: str, limits: RunLimits) -> None:
        self._connection.execute(
            """
            INSERT INTO agent_runs (
                run_id, question, status, answer, iterations, model_calls, limits_json
            ) VALUES (?, ?, 'running', NULL, 0, 0, ?)
            """,
            (run_id, question, limits.model_dump_json()),
        )
        self._connection.commit()

    def append_events(self, run_id: str, events: Sequence[TraceEvent]) -> None:
        self._connection.executemany(
            """
            INSERT INTO trace_events (
                run_id, sequence, node, event_type, iteration, details_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    sequence,
                    event.node,
                    event.event_type,
                    event.iteration,
                    json.dumps(event.details, sort_keys=True),
                )
                for sequence, event in enumerate(events, start=1)
            ],
        )
        self._connection.commit()

    def complete_run(self, result: AnswerResult) -> None:
        self._connection.execute(
            """
            UPDATE agent_runs
            SET status = ?, answer = ?, iterations = ?, model_calls = ?,
                result_json = ?, completed_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
            """,
            (
                result.status,
                result.answer,
                result.iterations,
                result.model_calls,
                result.model_dump_json(),
                result.run_id,
            ),
        )
        self._connection.commit()

    def read_run(self, run_id: str) -> dict[str, object]:
        run = self._connection.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if run is None:
            raise LookupError(f"Unknown run: {run_id}")
        events = self._connection.execute(
            "SELECT * FROM trace_events WHERE run_id = ? ORDER BY sequence", (run_id,)
        ).fetchall()
        return {
            "run": dict(run),
            "events": [dict(event) for event in events],
        }

    def close(self) -> None:
        self._connection.close()
