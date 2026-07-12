# comm_worker.py
import threading
import queue
import time
import requests
import concurrent.futures
import collections
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class PendingRequest:
    tag: Optional[str]
    url: str
    payload: Any
    headers: Dict[str, str]
    future: concurrent.futures.Future
    token_count: Optional[int] = None
    cancelled: bool = False
    inflight: bool = False  # 标记是否已被工作线程取出并开始发送

    @property
    def payload_size(self) -> int:
        payload = self.payload
        if isinstance(payload, (bytes, bytearray)):
            return len(payload)
        try:
            return len(payload)
        except Exception:
            return 0

class BandwidthSender:
    def __init__(
        self,
        bandwidth_MBps: float,
        base_latency: float = 0.0,
        timeout: int = 10,
        use_env_proxy: bool = False,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._q = queue.Queue()
        self._bandwidth_bytes = bandwidth_MBps * 1_000_000
        self._base_latency = base_latency
        self._timeout = timeout
        self._use_env_proxy = use_env_proxy
        self._on_complete = on_complete
        self._lock = threading.Lock()
        self._pending_requests = collections.defaultdict(collections.deque)
        self._tag_futures = collections.defaultdict(list)
        self._future_to_request = {}
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def set_bandwidth(self, bandwidth_MBps: float) -> None:
        with self._lock:
            self._bandwidth_bytes = bandwidth_MBps * 1_000_000

    def submit(self, url, payload, headers=None, tag=None, token_count=None):
        fut = concurrent.futures.Future()
        request = PendingRequest(
            tag=tag,
            url=url,
            payload=payload,
            headers=headers or {},
            future=fut,
            token_count=token_count,
        )
        with self._lock:
            self._future_to_request[fut] = request
            if tag is not None:
                self._pending_requests[tag].append(request)
                self._tag_futures[tag].append(fut)
        self._q.put(request)
        return fut

    def submit_many(self, url, payloads, headers=None, tag=None):
        """批量提交多个 payload，返回对应 Future 列表。"""
        futures = []
        for payload in payloads:
            futures.append(self.submit(url, payload, headers=headers, tag=tag))
        return futures

    def cancel_future(self, fut):
        with self._lock:
            request = self._future_to_request.get(fut)
            if request is not None:
                request.cancelled = True
        fut.cancel()

    def cancel_tag(self, tag):
        return len(self.drain_tag(tag))

    def drain_tag(self, tag) -> List[PendingRequest]:
        """取消并收集尚未发送(非 inflight)的指定 tag 请求。
        已经被工作线程取出的 inflight 请求不会被取消，也不会从 pending 队列移除，
        它们会在发送完成后由 _release_pending 正常清理。
        """
        cancelled_items: List[PendingRequest] = []
        with self._lock:
            pending = self._pending_requests.get(tag)
            if not pending:
                return []
            kept = collections.deque()
            for req in list(pending):
                # 仅取消未开始发送的任务
                if (not req.inflight) and (not req.future.done()) and (not req.cancelled):
                    cancelled_items.append(req)
                    # 从映射中移除 future
                    futs = self._tag_futures.get(tag)
                    if futs:
                        try:
                            futs.remove(req.future)
                        except ValueError:
                            pass
                        if not futs:
                            self._tag_futures.pop(tag, None)
                    self._future_to_request.pop(req.future, None)
                else:
                    kept.append(req)
            # 更新队列，保留未被取消或已在飞的请求
            if kept:
                self._pending_requests[tag] = kept
            else:
                self._pending_requests.pop(tag, None)
        # 在锁外标记和取消 future
        for req in cancelled_items:
            req.cancelled = True
            req.future.cancel()
        return cancelled_items

    def cancel_and_collect(self, tag) -> List[PendingRequest]:
        """兼容接口，等价于 drain_tag。"""
        return self.drain_tag(tag)

    def cancel_and_resubmit(self, tag, url, payload, headers=None, new_tag=None):
        """取消原 tag 的挂起请求并提交新的 payload。"""
        self.drain_tag(tag)
        return self.submit(url, payload, headers=headers, tag=new_tag or tag)

    def is_inflight_future(self, fut) -> bool:
        """查询指定 Future 是否对应一个已出队正在发送的请求。
        注意：请求完成后会从映射中移除，此时返回 False；配合同步的
        fut.done()/fut.cancelled() 判断更可靠。
        """
        with self._lock:
            req = self._future_to_request.get(fut)
            if req is None:
                return False
            return bool(getattr(req, "inflight", False)) and (not fut.done()) and (not fut.cancelled())

    def list_tags(self):
        """返回当前队列里还未发送完成的 tag -> 未完成任务数"""
        with self._lock:
            summary = {}
            for tag, requests_queue in self._pending_requests.items():
                remaining = sum(1 for req in requests_queue if not req.future.done())
                if remaining:
                    summary[tag] = remaining
            return summary

    def close(self):
        self._q.put(None)
        self._thread.join()

    def _release_pending(self, request: PendingRequest):
        tag = request.tag
        with self._lock:
            if tag is not None:
                pending = self._pending_requests.get(tag)
                if pending:
                    try:
                        pending.remove(request)
                    except ValueError:
                        pass
                    if not pending:
                        self._pending_requests.pop(tag, None)
                futures = self._tag_futures.get(tag)
                if futures:
                    try:
                        futures.remove(request.future)
                    except ValueError:
                        pass
                    if not futures:
                        self._tag_futures.pop(tag, None)
            self._future_to_request.pop(request.future, None)

    def _notify_complete(self, request: PendingRequest, started_at: float, finished_at: float, success: bool):
        if self._on_complete is None:
            return
        try:
            self._on_complete({
                "tag": request.tag,
                "url": request.url,
                "payload_size": request.payload_size,
                "token_count": request.token_count,
                "elapsed_seconds": max(0.0, finished_at - started_at),
                "success": success,
            })
        except Exception:
            pass

    def _worker(self):
        session = requests.Session()
        session.trust_env = self._use_env_proxy
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                request: PendingRequest = item

                # 取出后立即标记 inflight，避免被 drain_tag 当作可取消
                with self._lock:
                    request.inflight = True

                if request.cancelled or request.future.cancelled():
                    self._release_pending(request)
                    continue

                if self._bandwidth_bytes > 0:
                    quota = (request.payload_size / self._bandwidth_bytes) + self._base_latency
                else:
                    quota = self._base_latency
                with self._lock:
                    now = time.monotonic()

                started_at = time.monotonic()
                success = False
                try:
                    if isinstance(request.payload, (bytes, bytearray)):
                        resp = session.post(
                            request.url,
                            data=request.payload,
                            headers=request.headers,
                            timeout=self._timeout,
                        )
                    else:
                        resp = session.post(
                            request.url,
                            json=request.payload,
                            headers=request.headers,
                            timeout=self._timeout,
                        )
                    if resp.status_code == 502:
                        if not request.future.cancelled():
                            request.future.set_result({"error": 502})
                        continue
                    resp.raise_for_status()
                    after = time.monotonic()
                    if after - now < quota:
                        time.sleep(quota - (after - now))
                    success = True
                    if not request.future.cancelled():
                        request.future.set_result(
                            resp.json()
                            if 'application/json' in resp.headers.get('Content-Type', '')
                            else resp.content
                        )
                except Exception as exc:
                    if not request.future.cancelled():
                        request.future.set_exception(exc)
                finally:
                    self._notify_complete(request, started_at, time.monotonic(), success)
                    self._release_pending(request)
        finally:
            session.close()

def reconstruct_tasks(sender: BandwidthSender, tag_prefix: str):
    """重组具有相同 tag_prefix 的任务的结果，按 tag 后缀的数字顺序排列"""
    results = []
    idx = 0
    while True:
        tag = f"{tag_prefix}_{idx}"
        with sender._lock:
            futs = list(sender._tag_futures.pop(tag, []))
        if not futs:
            break
        for fut in futs:
            results.append(fut.result())
        idx += 1
    return results
