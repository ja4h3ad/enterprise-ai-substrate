"""Privacy-safe OpenTelemetry instrumentation for local ContractGraph runs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, Status, StatusCode


def content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RecordedSpan:
    name: str
    attributes: dict[str, Any]
    status: str


class TelemetryRecorder:
    """Application-owned SDK provider; exporters remain replaceable in production."""

    def __init__(self) -> None:
        self._exporter = InMemorySpanExporter()
        self._provider = TracerProvider(
            resource=Resource.create({"service.name": "contractgraph"})
        )
        self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        self._tracer = self._provider.get_tracer("contractgraph.application", "0.1.0")

    @contextmanager
    def span(
        self, name: str, attributes: Mapping[str, Any]
    ) -> Iterator[Span]:
        with self._tracer.start_as_current_span(name, attributes=dict(attributes)) as span:
            try:
                yield span
            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, type(error).__name__))
                span.set_attribute("error.type", type(error).__name__)
                raise

    def spans_for_run(self, run_id: str) -> tuple[RecordedSpan, ...]:
        spans = []
        for span in self._exporter.get_finished_spans():
            attributes = dict(span.attributes or {})
            if attributes.get("contractgraph.run.id") != run_id:
                continue
            spans.append(
                RecordedSpan(
                    name=span.name,
                    attributes=attributes,
                    status=span.status.status_code.name,
                )
            )
        return tuple(spans)

    @staticmethod
    def mark_error(span: Span, error_type: str) -> None:
        span.set_status(Status(StatusCode.ERROR, error_type))
        span.set_attribute("error.type", error_type)

    def shutdown(self) -> None:
        self._provider.shutdown()
