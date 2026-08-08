"""Fair micro-batch scheduler for one multi-sequence target-model worker."""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
import threading
import time
from typing import Callable, Deque, Dict, List, Optional

from .batch_backend import VerifyRequest


@dataclass
class _WorkItem:
    kind: str
    task_id: int
    payload: object = None
    enqueued_at: float = field(default_factory=time.perf_counter)
    event: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: Optional[BaseException] = None


class _BatchEnergySink:
    task_id = "multi-sequence-batch"

    def __init__(self) -> None:
        self.total_gpu_power_integral_joules = 0.0
        self.measurement: Dict[str, object] = {}

    def record_energy_measurement(
        self, *, stage, energy_joules, duration_seconds, sample_count
    ) -> None:
        self.total_gpu_power_integral_joules = float(energy_joules)
        self.measurement = {
            "stage": stage,
            "energy_joules": float(energy_joules),
            "duration_seconds": float(duration_seconds),
            "sample_count": int(sample_count),
        }


class VerificationBatchScheduler:
    def __init__(
        self,
        backend,
        *,
        max_batch_clients: int = 8,
        max_batch_tokens: int = 1024,
        batch_wait_ms: float = 2.0,
        request_timeout_s: float = 300.0,
        energy_context_factory: Optional[Callable[[object, str], object]] = None,
    ) -> None:
        self.backend = backend
        self.max_batch_clients = max(1, int(max_batch_clients))
        self.max_batch_tokens = max(1, int(max_batch_tokens))
        self.batch_wait_seconds = max(0.0, float(batch_wait_ms) / 1000.0)
        self.request_timeout_s = max(1.0, float(request_timeout_s))
        self.energy_context_factory = energy_context_factory
        self._condition = threading.Condition()
        self._queue: Deque[_WorkItem] = deque()
        self._stopping = False
        self._batch_id = 0
        self._stats = {
            "batches": 0,
            "requests": 0,
            "prefill_batches": 0,
            "prefill_requests": 0,
            "evaluated_tokens": 0,
            "total_queue_seconds": 0.0,
            "total_decode_seconds": 0.0,
            "max_actual_batch_clients": 0,
            "max_actual_batch_tokens": 0,
            "max_actual_prefill_batch_clients": 0,
        }
        self._worker = threading.Thread(
            target=self._run, name="pipesd-batch-worker", daemon=True
        )
        self._worker.start()

    def _enqueue_and_wait(self, item: _WorkItem):
        with self._condition:
            if self._stopping:
                raise RuntimeError("batch scheduler is stopping")
            self._queue.append(item)
            self._condition.notify_all()
        if not item.event.wait(self.request_timeout_s):
            raise TimeoutError(
                f"cloud batch scheduler timed out after {self.request_timeout_s}s"
            )
        if item.error is not None:
            raise item.error
        return item.result

    def initialize(self, task_id: int, prefix: List[int]):
        return self._enqueue_and_wait(_WorkItem("init", task_id, list(prefix)))

    def submit(self, request: VerifyRequest):
        return self._enqueue_and_wait(_WorkItem("verify", request.task_id, request))

    def resolve_rejection(self, task_id: int, verification_id: str, draft_probs):
        return self._enqueue_and_wait(
            _WorkItem("resolve", task_id, (verification_id, draft_probs))
        )

    def close_session(self, task_id: int) -> None:
        self._enqueue_and_wait(_WorkItem("close", task_id))

    def shutdown(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._worker.join(timeout=10.0)
        self.backend.close()

    def _next_item(self) -> Optional[_WorkItem]:
        with self._condition:
            while not self._queue and not self._stopping:
                self._condition.wait()
            if not self._queue:
                return None
            return self._queue.popleft()

    def _collect_verify_batch(self, first: _WorkItem) -> List[_WorkItem]:
        items = [first]
        task_ids = {first.task_id}
        token_count = self.backend.estimate_input_tokens(first.payload)
        deadline = first.enqueued_at + self.batch_wait_seconds
        while len(items) < self.max_batch_clients and token_count < self.max_batch_tokens:
            with self._condition:
                while not self._queue and not self._stopping:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        return items
                    self._condition.wait(remaining)
                if not self._queue:
                    return items
                selected_index = None
                selected_tokens = 0
                for index, candidate in enumerate(self._queue):
                    if candidate.kind != "verify" or candidate.task_id in task_ids:
                        continue
                    candidate_tokens = self.backend.estimate_input_tokens(candidate.payload)
                    if token_count + candidate_tokens <= self.max_batch_tokens:
                        selected_index = index
                        selected_tokens = candidate_tokens
                        break
                if selected_index is None:
                    if time.perf_counter() >= deadline:
                        return items
                    self._condition.wait(max(0.0, deadline - time.perf_counter()))
                    continue
                candidate = self._queue[selected_index]
                del self._queue[selected_index]
            items.append(candidate)
            task_ids.add(candidate.task_id)
            token_count += selected_tokens
        return items

    def _collect_init_batch(self, first: _WorkItem) -> List[_WorkItem]:
        items = [first]
        task_ids = {first.task_id}
        deadline = first.enqueued_at + self.batch_wait_seconds
        while len(items) < self.max_batch_clients:
            with self._condition:
                while not self._queue and not self._stopping:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        return items
                    self._condition.wait(remaining)
                selected_index = next(
                    (
                        index for index, candidate in enumerate(self._queue)
                        if candidate.kind == 'init'
                        and candidate.task_id not in task_ids
                    ),
                    None,
                )
                if selected_index is None:
                    return items
                candidate = self._queue[selected_index]
                del self._queue[selected_index]
            items.append(candidate)
            task_ids.add(candidate.task_id)
        return items

    @staticmethod
    def _finish(item: _WorkItem, *, result=None, error=None) -> None:
        item.result = result
        item.error = error
        item.event.set()

    def _run_non_verify(self, item: _WorkItem) -> None:
        try:
            if item.kind == "close":
                self.backend.close_session(item.task_id)
                result = None
            elif item.kind == "resolve":
                verification_id, draft_probs = item.payload
                result = self.backend.resolve_rejection(
                    item.task_id, verification_id, draft_probs
                )
            else:
                raise RuntimeError(f"unknown batch work item: {item.kind}")
            self._finish(item, result=result)
        except BaseException as exc:
            self._finish(item, error=exc)

    def _run_init(self, items: List[_WorkItem]) -> None:
        batch_id = self._batch_id
        self._batch_id += 1
        sink = _BatchEnergySink()
        context = (
            self.energy_context_factory(sink, "prompt_prefill")
            if self.energy_context_factory is not None
            else nullcontext()
        )
        started_at = time.perf_counter()
        try:
            with context:
                results = self.backend.initialize_sessions(
                    [(item.task_id, item.payload) for item in items]
                )
            decode_seconds = time.perf_counter() - started_at
            by_task = {int(result['task_id']): result for result in results}
            total_tokens = sum(
                int(result.get('evaluated_tokens', 0)) for result in results
            )
            batch_energy = float(sink.measurement.get('energy_joules', 0.0) or 0.0)
            for item in items:
                result = dict(by_task[item.task_id])
                evaluated_tokens = int(result.get('evaluated_tokens', 0))
                result.update({
                    'batch_stage': 'prefill',
                    'batch_id': batch_id,
                    'actual_batch_size': len(items),
                    'actual_batch_tokens': total_tokens,
                    'prefill_seconds': decode_seconds,
                    'batch_queue_seconds': max(0.0, started_at - item.enqueued_at),
                    'prefill_gpu_power_integral': (
                        batch_energy * evaluated_tokens / total_tokens
                        if total_tokens > 0 else 0.0
                    ),
                    'batch_energy_joules': batch_energy,
                    'energy_allocation': 'evaluated_token_share',
                })
                self._finish(item, result=result)
            self._stats['prefill_batches'] += 1
            self._stats['prefill_requests'] += len(items)
            self._stats['max_actual_prefill_batch_clients'] = max(
                self._stats['max_actual_prefill_batch_clients'], len(items)
            )
        except BaseException as exc:
            for item in items:
                self._finish(item, error=exc)

    def _run_verify(self, items: List[_WorkItem]) -> None:
        batch_id = self._batch_id
        self._batch_id += 1
        sink = _BatchEnergySink()
        context = (
            self.energy_context_factory(sink, "verify_total")
            if self.energy_context_factory is not None
            else nullcontext()
        )
        started_at = time.perf_counter()
        try:
            with context:
                results = self.backend.verify_batch([item.payload for item in items])
            decode_seconds = time.perf_counter() - started_at
            by_task = {int(result["task_id"]): result for result in results}
            total_tokens = sum(int(result.get("evaluated_tokens", 0)) for result in results)
            batch_energy = float(sink.measurement.get("energy_joules", 0.0) or 0.0)
            for item in items:
                result = dict(by_task[item.task_id])
                evaluated_tokens = int(result.get("evaluated_tokens", 0))
                allocated_energy = (
                    batch_energy * evaluated_tokens / total_tokens if total_tokens > 0 else 0.0
                )
                result.update(
                    {
                        "batch_id": batch_id,
                        "batch_stage": "verify",
                        "actual_batch_size": len(items),
                        "actual_batch_tokens": total_tokens,
                        "batch_queue_seconds": max(0.0, started_at - item.enqueued_at),
                        "batch_decode_seconds": decode_seconds,
                        "batch_energy_joules": batch_energy,
                        "gpu_power_integral": allocated_energy,
                        "energy_allocation": "evaluated_token_share",
                    }
                )
                self._finish(item, result=result)
            self._stats["batches"] += 1
            self._stats["requests"] += len(items)
            self._stats["evaluated_tokens"] += total_tokens
            self._stats["total_queue_seconds"] += sum(
                max(0.0, started_at - item.enqueued_at) for item in items
            )
            self._stats["total_decode_seconds"] += decode_seconds
            self._stats["max_actual_batch_clients"] = max(
                self._stats["max_actual_batch_clients"], len(items)
            )
            self._stats["max_actual_batch_tokens"] = max(
                self._stats["max_actual_batch_tokens"], total_tokens
            )
        except BaseException as exc:
            for item in items:
                self._finish(item, error=exc)

    def _run(self) -> None:
        while True:
            item = self._next_item()
            if item is None:
                return
            if item.kind == "init":
                self._run_init(self._collect_init_batch(item))
                continue
            if item.kind != "verify":
                self._run_non_verify(item)
                continue
            self._run_verify(self._collect_verify_batch(item))

    def snapshot(self) -> Dict[str, object]:
        with self._condition:
            queued = len(self._queue)
        stats = dict(self._stats)
        requests = int(stats["requests"])
        batches = int(stats["batches"])
        stats.update(
            {
                "queue_depth": queued,
                "max_batch_clients": self.max_batch_clients,
                "max_batch_tokens": self.max_batch_tokens,
                "batch_wait_ms": self.batch_wait_seconds * 1000.0,
                "mean_actual_batch_size": requests / batches if batches else 0.0,
                "mean_queue_seconds": (
                    float(stats["total_queue_seconds"]) / requests if requests else 0.0
                ),
                "backend": self.backend.snapshot(),
            }
        )
        return stats
