"""Multi-sequence llama.cpp backend for batched PipeSD verification.

The high-level ``llama_cpp.Llama.eval`` API owns sequence 0 and cannot batch
independent client KV histories.  This module intentionally uses the public
low-level ctypes bindings so one ``llama_decode`` call can contain tokens from
multiple sequence ids.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import threading
import uuid
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class VerifyRequest:
    task_id: int
    n_past: int
    tokens: List[int]
    draft_probs: List[np.ndarray]
    seed: int
    temp: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    prob_transport: str = "full"


@dataclass
class _Session:
    task_id: int
    seq_id: int
    kv_tokens: List[int]
    next_logits: np.ndarray
    pending_final_token: Optional[int] = None
    pending_rejection: Optional[Dict[str, object]] = None

    @property
    def kv_n_past(self) -> int:
        return len(self.kv_tokens)

    @property
    def logical_n_past(self) -> int:
        return self.kv_n_past + (1 if self.pending_final_token is not None else 0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


def _positive_distribution(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(clipped.sum())
    if total <= 0:
        return np.full(clipped.shape, 1.0 / clipped.size, dtype=np.float64)
    return clipped / total


class LlamaCppBatchBackend:
    """One target model/context with a distinct llama.cpp sequence per task."""

    name = "llama_cpp_multi_sequence"

    def __init__(
        self,
        *,
        model_path: str,
        max_sequences: int = 8,
        context_tokens_per_sequence: int = 1024,
        decode_batch_tokens: int = 1024,
        physical_batch_tokens: int = 64,
        threads: int = 4,
        flash_attention: bool = True,
    ) -> None:
        if max_sequences <= 0:
            raise ValueError("max_sequences must be positive")
        if context_tokens_per_sequence <= 0:
            raise ValueError("context_tokens_per_sequence must be positive")
        if decode_batch_tokens <= 0:
            raise ValueError("decode_batch_tokens must be positive")

        try:
            from llama_cpp import llama_cpp as lib  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised on the GPU host
            raise RuntimeError(
                "batched backend requires a CUDA-enabled llama-cpp-python build"
            ) from exc

        self._lib = lib
        self.max_sequences = int(max_sequences)
        self.context_tokens_per_sequence = int(context_tokens_per_sequence)
        self.decode_batch_tokens = int(decode_batch_tokens)
        self._lock = threading.RLock()
        self._sessions: Dict[int, _Session] = {}
        self._free_seq_ids = list(range(self.max_sequences))
        self._closed = False

        self._require_symbols(
            "llama_model_default_params",
            "llama_context_default_params",
            "llama_batch_init",
            "llama_batch_free",
            "llama_decode",
            "llama_get_logits_ith",
        )
        self._backend_init()

        model_params = lib.llama_model_default_params()
        if hasattr(model_params, "n_gpu_layers"):
            model_params.n_gpu_layers = -1
        if hasattr(model_params, "use_mmap"):
            model_params.use_mmap = True
        if hasattr(model_params, "use_mlock"):
            model_params.use_mlock = False

        load_model = getattr(lib, "llama_model_load_from_file", None) or getattr(
            lib, "llama_load_model_from_file", None
        )
        if load_model is None:
            raise RuntimeError("llama-cpp-python is missing the model load API")
        self._model = load_model(str(model_path).encode("utf-8"), model_params)
        if not self._model:
            raise RuntimeError(f"failed to load target model: {model_path}")

        context_params = lib.llama_context_default_params()
        total_context = self.context_tokens_per_sequence * self.max_sequences
        self._set_struct_field(context_params, "n_ctx", total_context, required=True)
        self._set_struct_field(
            context_params, "n_batch", self.decode_batch_tokens, required=True
        )
        self._set_struct_field(
            context_params,
            "n_ubatch",
            min(int(physical_batch_tokens), self.decode_batch_tokens),
            required=False,
        )
        self._set_struct_field(
            context_params, "n_seq_max", self.max_sequences, required=True
        )
        self._set_struct_field(context_params, "n_threads", int(threads), required=False)
        self._set_struct_field(
            context_params, "n_threads_batch", int(threads), required=False
        )
        self._set_struct_field(context_params, "kv_unified", True, required=False)
        self._set_struct_field(
            context_params, "flash_attn", bool(flash_attention), required=False
        )

        init_context = getattr(lib, "llama_init_from_model", None) or getattr(
            lib, "llama_new_context_with_model", None
        )
        if init_context is None:
            raise RuntimeError("llama-cpp-python is missing the context init API")
        self._ctx = init_context(self._model, context_params)
        if not self._ctx:
            self._free_model()
            raise RuntimeError(
                "failed to create multi-sequence context; reduce context size or batch size"
            )

        set_threads = getattr(lib, "llama_set_n_threads", None)
        if set_threads is not None:
            set_threads(self._ctx, int(threads), int(threads))
        set_causal = getattr(lib, "llama_set_causal_attn", None)
        if set_causal is not None:
            set_causal(self._ctx, True)

        vocab = getattr(lib, "llama_model_get_vocab", None)
        vocab_size = None
        if vocab is not None and hasattr(lib, "llama_vocab_n_tokens"):
            vocab_size = int(lib.llama_vocab_n_tokens(vocab(self._model)))
        elif hasattr(lib, "llama_n_vocab"):
            vocab_size = int(lib.llama_n_vocab(self._model))
        if not vocab_size or vocab_size <= 0:
            self.close()
            raise RuntimeError("could not determine target model vocabulary size")
        self.vocab_size = vocab_size

    def _require_symbols(self, *names: str) -> None:
        missing = [name for name in names if not hasattr(self._lib, name)]
        if missing:
            raise RuntimeError(
                "llama-cpp-python does not support multi-sequence batching; "
                f"missing: {', '.join(missing)}"
            )

    @staticmethod
    def _set_struct_field(struct, name: str, value, *, required: bool) -> None:
        if hasattr(struct, name):
            setattr(struct, name, value)
        elif required:
            raise RuntimeError(
                f"installed llama-cpp-python context params lack required field {name}"
            )

    def _backend_init(self) -> None:
        init = getattr(self._lib, "llama_backend_init", None)
        if init is None:
            raise RuntimeError("llama-cpp-python is missing llama_backend_init")
        try:
            init()
        except TypeError:  # compatibility with older bindings
            init(False)

    def _free_model(self) -> None:
        if getattr(self, "_model", None):
            free_model = getattr(self._lib, "llama_model_free", None) or getattr(
                self._lib, "llama_free_model", None
            )
            if free_model is not None:
                free_model(self._model)
            self._model = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if getattr(self, "_ctx", None):
                self._lib.llama_free(self._ctx)
                self._ctx = None
            self._free_model()
            self._closed = True

    def _memory_seq_rm(self, seq_id: int, p0: int, p1: int = -1) -> bool:
        lib = self._lib
        if hasattr(lib, "llama_get_memory") and hasattr(lib, "llama_memory_seq_rm"):
            memory = lib.llama_get_memory(self._ctx)
            return bool(lib.llama_memory_seq_rm(memory, seq_id, p0, p1))
        if hasattr(lib, "llama_kv_cache_seq_rm"):
            return bool(lib.llama_kv_cache_seq_rm(self._ctx, seq_id, p0, p1))
        raise RuntimeError("llama-cpp-python lacks sequence KV removal support")

    def _decode_rows(self, rows: Sequence[tuple[int, int, int]]) -> List[np.ndarray]:
        """Decode ``(token, position, seq_id)`` rows in one llama_decode call."""
        if not rows:
            return []
        if len(rows) > self.decode_batch_tokens:
            raise ValueError(
                f"decode batch has {len(rows)} tokens, limit is {self.decode_batch_tokens}"
            )
        batch = self._lib.llama_batch_init(len(rows), 0, 1)
        try:
            batch.n_tokens = len(rows)
            for index, (token, position, seq_id) in enumerate(rows):
                batch.token[index] = int(token)
                batch.pos[index] = int(position)
                batch.n_seq_id[index] = 1
                batch.seq_id[index][0] = int(seq_id)
                batch.logits[index] = 1
            code = int(self._lib.llama_decode(self._ctx, batch))
            if code != 0:
                raise RuntimeError(f"llama_decode failed with code {code}")
            outputs: List[np.ndarray] = []
            for index in range(len(rows)):
                pointer = self._lib.llama_get_logits_ith(self._ctx, index)
                if not pointer:
                    raise RuntimeError(f"missing logits for batch row {index}")
                outputs.append(
                    np.ctypeslib.as_array(pointer, shape=(self.vocab_size,)).copy()
                )
            return outputs
        finally:
            self._lib.llama_batch_free(batch)

    def _decode_single_sequence(
        self, *, seq_id: int, tokens: Sequence[int], start_position: int
    ) -> np.ndarray:
        if not tokens:
            raise ValueError("cannot decode an empty sequence")
        last_logits = None
        for offset in range(0, len(tokens), self.decode_batch_tokens):
            chunk = tokens[offset : offset + self.decode_batch_tokens]
            rows = [
                (int(token), start_position + offset + index, seq_id)
                for index, token in enumerate(chunk)
            ]
            last_logits = self._decode_rows(rows)[-1]
        assert last_logits is not None
        return last_logits

    def initialize_sessions(
        self, requests: Sequence[tuple[int, Sequence[int]]]
    ) -> List[Dict[str, object]]:
        with self._lock:
            task_ids = [int(task_id) for task_id, _ in requests]
            if len(set(task_ids)) != len(task_ids):
                raise ValueError("duplicate task id in prefill batch")
            prefixes = {
                int(task_id): [int(token) for token in prefix]
                for task_id, prefix in requests
            }
            for task_id, prefix in prefixes.items():
                if not prefix:
                    raise ValueError("prefix must contain at least one token")
                if len(prefix) > self.context_tokens_per_sequence:
                    raise ValueError(
                        f"task {task_id} prefix exceeds per-sequence context capacity"
                    )
                if task_id in self._sessions:
                    self.close_session(task_id)
            if len(self._free_seq_ids) < len(requests):
                raise RuntimeError(
                    f"need {len(requests)} sequence slots but only "
                    f"{len(self._free_seq_ids)} are free"
                )
            seq_ids = {
                task_id: self._free_seq_ids.pop(0) for task_id in task_ids
            }
            for seq_id in seq_ids.values():
                self._memory_seq_rm(seq_id, 0, -1)
            try:
                cursors = {task_id: 0 for task_id in task_ids}
                next_logits: Dict[int, np.ndarray] = {}
                while any(cursors[task_id] < len(prefixes[task_id]) for task_id in task_ids):
                    rows = []
                    owners = []
                    while len(rows) < self.decode_batch_tokens:
                        progressed = False
                        for task_id in task_ids:
                            cursor = cursors[task_id]
                            if cursor >= len(prefixes[task_id]):
                                continue
                            rows.append(
                                (
                                    prefixes[task_id][cursor],
                                    cursor,
                                    seq_ids[task_id],
                                )
                            )
                            owners.append(task_id)
                            cursors[task_id] += 1
                            progressed = True
                            if len(rows) >= self.decode_batch_tokens:
                                break
                        if not progressed:
                            break
                    for task_id, logits in zip(owners, self._decode_rows(rows)):
                        next_logits[task_id] = logits
            except Exception:
                for seq_id in seq_ids.values():
                    self._memory_seq_rm(seq_id, 0, -1)
                    self._free_seq_ids.append(seq_id)
                self._free_seq_ids.sort()
                raise
            results = []
            for task_id in task_ids:
                self._sessions[task_id] = _Session(
                    task_id=task_id,
                    seq_id=seq_ids[task_id],
                    kv_tokens=prefixes[task_id],
                    next_logits=next_logits[task_id],
                )
                results.append(
                    {
                        "task_id": task_id,
                        "n_past": len(prefixes[task_id]),
                        "seq_id": seq_ids[task_id],
                        "evaluated_tokens": len(prefixes[task_id]),
                    }
                )
            return results

    def initialize_session(self, task_id: int, prefix: Sequence[int]) -> Dict[str, object]:
        return self.initialize_sessions([(task_id, prefix)])[0]

    def close_session(self, task_id: int) -> None:
        with self._lock:
            session = self._sessions.pop(task_id, None)
            if session is None:
                return
            self._memory_seq_rm(session.seq_id, 0, -1)
            self._free_seq_ids.append(session.seq_id)
            self._free_seq_ids.sort()

    def estimate_input_tokens(self, request: VerifyRequest) -> int:
        session = self._sessions.get(request.task_id)
        pending = bool(session and session.pending_final_token is not None)
        return len(request.tokens) + int(pending)

    def verify_batch(self, requests: Sequence[VerifyRequest]) -> List[Dict[str, object]]:
        if not requests:
            return []
        with self._lock:
            rows: List[tuple[int, int, int]] = []
            row_ranges: Dict[int, tuple[int, int, List[int]]] = {}
            for request in requests:
                session = self._sessions.get(request.task_id)
                if session is None:
                    raise RuntimeError(f"task {request.task_id} has no batch session")
                if request.temp != 0.0:
                    raise ValueError("batched backend currently requires temperature 0")
                if session.pending_rejection is not None:
                    raise RuntimeError(
                        f"task {request.task_id} must resolve its pending rejection first"
                    )
                if request.n_past != session.logical_n_past:
                    raise RuntimeError(
                        f"task {request.task_id} n_past mismatch: edge={request.n_past}, "
                        f"cloud={session.logical_n_past}"
                    )
                if len(request.tokens) != len(request.draft_probs):
                    raise ValueError("draft token/probability lengths differ")
                if not request.tokens:
                    raise ValueError("verification request contains no draft tokens")
                inputs = (
                    [session.pending_final_token]
                    if session.pending_final_token is not None
                    else []
                ) + [int(token) for token in request.tokens]
                if session.kv_n_past + len(inputs) > self.context_tokens_per_sequence:
                    raise RuntimeError(
                        f"task {request.task_id} exceeds per-sequence context capacity"
                    )
                start = len(rows)
                rows.extend(
                    (token, session.kv_n_past + index, session.seq_id)
                    for index, token in enumerate(inputs)
                )
                row_ranges[request.task_id] = (start, len(rows), inputs)

            if len(rows) > self.decode_batch_tokens:
                raise RuntimeError(
                    f"scheduler built {len(rows)} tokens but backend limit is "
                    f"{self.decode_batch_tokens}"
                )
            decoded_logits = self._decode_rows(rows)
            results: List[Dict[str, object]] = []
            for request in requests:
                session = self._sessions[request.task_id]
                row_start, row_end, inputs = row_ranges[request.task_id]
                output_logits = decoded_logits[row_start:row_end]
                had_pending = session.pending_final_token is not None
                draft_logits: List[np.ndarray] = []
                if had_pending:
                    draft_logits.append(output_logits[0])
                    draft_logits.extend(output_logits[1:-1])
                else:
                    draft_logits.append(session.next_logits)
                    draft_logits.extend(output_logits[:-1])
                if len(draft_logits) != len(request.tokens):
                    raise RuntimeError("internal logits alignment error")

                target_probs = np.stack([_softmax(row) for row in draft_logits])
                draft_tokens = np.asarray(request.tokens, dtype=np.int64)
                token_rows = np.arange(len(request.tokens))
                if request.prob_transport == "lazy_distribution":
                    draft_token_probs = np.asarray(
                        [float(np.asarray(row)) for row in request.draft_probs],
                        dtype=np.float64,
                    )
                    draft_probs = None
                elif request.prob_transport == "full":
                    draft_probs = np.stack(
                        [np.asarray(row, dtype=np.float64) for row in request.draft_probs]
                    )
                    if draft_probs.shape != target_probs.shape:
                        raise ValueError(
                            f"draft probability shape {draft_probs.shape} does not match "
                            f"target shape {target_probs.shape}"
                        )
                    draft_token_probs = draft_probs[token_rows, draft_tokens]
                else:
                    raise ValueError(
                        f"unknown probability transport: {request.prob_transport}"
                    )
                ratios = target_probs[token_rows, draft_tokens] / (
                    draft_token_probs + 1e-9
                )
                accepted = 0
                for index, ratio in enumerate(ratios):
                    rand_value = random.Random(
                        request.seed + request.n_past + index
                    ).random()
                    if ratio >= 1.0 or rand_value < float(ratio):
                        accepted += 1
                    else:
                        break

                if accepted < len(request.tokens):
                    if request.prob_transport == "lazy_distribution":
                        verification_id = uuid.uuid4().hex
                        session.pending_rejection = {
                            "verification_id": verification_id,
                            "target_probs": target_probs[accepted].copy(),
                            "draft_token_prob": float(draft_token_probs[accepted]),
                            "draft_token": int(request.tokens[accepted]),
                            "n_accepted": accepted,
                            "n_speculative": len(request.tokens),
                            "seed": int(request.seed),
                        }
                        final_token = None
                    else:
                        final_distribution = _positive_distribution(
                            target_probs[accepted] - draft_probs[accepted]
                        )
                        final_token = int(
                            np.random.default_rng(request.seed).choice(
                                final_distribution.size, p=final_distribution
                            )
                        )
                else:
                    final_token = int(np.argmax(output_logits[-1]))

                committed_inputs = int(had_pending) + accepted
                rollback_position = session.kv_n_past + committed_inputs
                if committed_inputs < len(inputs):
                    if not self._memory_seq_rm(session.seq_id, rollback_position, -1):
                        raise RuntimeError(
                            f"llama.cpp could not roll back task {request.task_id} KV suffix"
                        )
                session.kv_tokens.extend(inputs[:committed_inputs])
                if committed_inputs > 0:
                    session.next_logits = output_logits[committed_inputs - 1]
                session.pending_final_token = final_token
                result = {
                        "task_id": request.task_id,
                        "n_accepted": accepted,
                        "n_speculative": len(request.tokens),
                        "final_token": final_token,
                        "n_past": session.logical_n_past,
                        "evaluated_tokens": len(inputs),
                        "seq_id": session.seq_id,
                    }
                if session.pending_rejection is not None:
                    result.update({
                        "status": "needs_full_probs",
                        "verification_id": session.pending_rejection["verification_id"],
                        "rejected_index": accepted,
                    })
                results.append(result)
            return results

    def resolve_rejection(
        self,
        task_id: int,
        verification_id: str,
        draft_probs: np.ndarray,
    ) -> Dict[str, object]:
        with self._lock:
            session = self._sessions.get(task_id)
            if session is None:
                raise RuntimeError(f"task {task_id} has no batch session")
            pending = session.pending_rejection
            if pending is None:
                raise RuntimeError(f"task {task_id} has no pending rejection")
            if pending["verification_id"] != verification_id:
                raise RuntimeError("verification_id does not match pending rejection")

            draft_probs = np.asarray(draft_probs, dtype=np.float64).reshape(-1)
            target_probs = np.asarray(pending["target_probs"], dtype=np.float64)
            if draft_probs.shape != target_probs.shape:
                raise ValueError(
                    f"draft probability shape {draft_probs.shape} does not match "
                    f"target shape {target_probs.shape}"
                )
            token = int(pending["draft_token"])
            if not np.isclose(
                draft_probs[token],
                float(pending["draft_token_prob"]),
                rtol=1e-5,
                atol=1e-7,
            ):
                raise ValueError("resolved draft probability does not match proposal scalar")

            final_distribution = _positive_distribution(target_probs - draft_probs)
            final_token = int(
                np.random.default_rng(int(pending["seed"])).choice(
                    final_distribution.size, p=final_distribution
                )
            )
            session.pending_final_token = final_token
            session.pending_rejection = None
            return {
                "task_id": task_id,
                "status": "resolved",
                "n_accepted": int(pending["n_accepted"]),
                "n_speculative": int(pending["n_speculative"]),
                "final_token": final_token,
                "n_past": session.logical_n_past,
                "evaluated_tokens": 0,
                "seq_id": session.seq_id,
            }

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "backend": self.name,
                "max_sequences": self.max_sequences,
                "active_sequences": len(self._sessions),
                "free_sequences": len(self._free_seq_ids),
                "context_tokens_per_sequence": self.context_tokens_per_sequence,
                "decode_batch_tokens": self.decode_batch_tokens,
            }
