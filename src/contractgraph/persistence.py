"""SQLite-backed checkpoint and domain-trace persistence seams."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol, Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from contractgraph.application_models import (
    AnalystDecision,
    AnswerResult,
    ReviewPacket,
    RunLimits,
    TraceEvent,
)


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

            CREATE TABLE IF NOT EXISTS analyst_reviews (
                run_id TEXT PRIMARY KEY REFERENCES agent_runs(run_id),
                status TEXT NOT NULL CHECK(status IN ('pending', 'resuming', 'resolved')),
                checkpoint_id TEXT NOT NULL,
                packet_json TEXT NOT NULL,
                decision_json TEXT,
                before_status TEXT NOT NULL,
                after_status TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT
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
        existing = self._connection.execute(
            "SELECT COUNT(*) FROM trace_events WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        new_events = events[existing:]
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
                for sequence, event in enumerate(new_events, start=existing + 1)
            ],
        )
        self._connection.commit()

    def pause_for_review(self, result: AnswerResult, packet: ReviewPacket) -> None:
        with self._connection:
            self._connection.execute(
                """
                UPDATE agent_runs
                SET status = ?, iterations = ?, model_calls = ?, result_json = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (
                    result.status,
                    result.iterations,
                    result.model_calls,
                    result.model_dump_json(),
                    result.run_id,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO analyst_reviews (
                    run_id, status, checkpoint_id, packet_json, before_status,
                    created_at
                ) VALUES (?, 'pending', ?, ?, 'review_required', ?)
                """,
                (
                    packet.run_id,
                    packet.checkpoint_id,
                    packet.model_dump_json(),
                    packet.created_at.isoformat(),
                ),
            )

    def read_review(self, run_id: str) -> ReviewPacket:
        row = self._connection.execute(
            "SELECT * FROM analyst_reviews WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise LookupError(f"No analyst review for run: {run_id}")
        packet = ReviewPacket.model_validate_json(row["packet_json"])
        return packet.model_copy(
            update={
                "status": row["status"],
                "resolved_at": row["resolved_at"],
            }
        )

    def begin_review_resolution(
        self, run_id: str, checkpoint_id: str, decision: AnalystDecision
    ) -> ReviewPacket:
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE analyst_reviews
                SET status = 'resuming', decision_json = ?
                WHERE run_id = ? AND checkpoint_id = ? AND status = 'pending'
                """,
                (decision.model_dump_json(), run_id, checkpoint_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("stale_or_already_resolved_checkpoint")
        return self.read_review(run_id)

    def complete_review_resolution(
        self, result: AnswerResult, resolved_at: str
    ) -> ReviewPacket:
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE analyst_reviews
                SET status = 'resolved', after_status = ?, resolved_at = ?
                WHERE run_id = ? AND status = 'resuming'
                """,
                (result.status, resolved_at, result.run_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("review_resolution_not_in_progress")
        return self.read_review(result.run_id)

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
