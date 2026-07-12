import argparse
import json
import math
import os
import statistics
import time
from typing import Dict, List

import requests


DEFAULT_URL = os.getenv("EDGE_HEALTH_URL", "http://127.0.0.1:8000/health")
# DEFAULT_URL = os.getenv("EDGE_HEALTH_URL", "http://i-2.gpushare.com:55057/health")


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)

    ordered = sorted(values)
    rank = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_latencies(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {
            "count": 0,
            "min_ms": None,
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }

    return {
        "count": len(latencies_ms),
        "min_ms": min(latencies_ms),
        "avg_ms": statistics.mean(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "max_ms": max(latencies_ms),
    }


def _measure_once(session: requests.Session, url: str, timeout: float) -> float:
    start = time.perf_counter()
    response = session.post(url, timeout=timeout)
    response.raise_for_status()
    end = time.perf_counter()
    print((end - start) * 1000.0)
    return (end - start) * 1000.0


def measure_rtt(
    url: str,
    count: int,
    timeout: float,
    warmup: int = 1,
    use_env_proxy: bool = False,
) -> Dict[str, object]:
    latencies_ms: List[float] = []
    errors: List[str] = []

    with requests.Session() as session:
        session.trust_env = use_env_proxy
        for _ in range(max(0, warmup)):
            _measure_once(session, url, timeout)

        for _ in range(count):
            try:
                latencies_ms.append(_measure_once(session, url, timeout))
            except Exception as exc:
                errors.append(str(exc))

    summary = summarize_latencies(latencies_ms)
    return {
        "url": url,
        "timeout_s": timeout,
        "warmup": warmup,
        "requested_count": count,
        "success_count": len(latencies_ms),
        "failure_count": len(errors),
        "latencies_ms": latencies_ms,
        "errors": errors,
        "summary": summary,
    }


def format_report(result: Dict[str, object]) -> str:
    summary = result["summary"]
    lines = [
        f"URL: {result['url']}",
        f"Requests: {result['requested_count']} (success={result['success_count']}, failure={result['failure_count']}, warmup={result['warmup']})",
        f"Timeout: {result['timeout_s']} s",
    ]

    if summary["count"] == 0:
        lines.append("No successful RTT samples collected.")
    else:
        lines.extend(
            [
                f"min/avg/p50/p95/max: "
                f"{summary['min_ms']:.2f} / {summary['avg_ms']:.2f} / "
                f"{summary['p50_ms']:.2f} / {summary['p95_ms']:.2f} / {summary['max_ms']:.2f} ms",
            ]
        )

    if result["errors"]:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in result["errors"])

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure HTTP RTT to the health endpoint.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Health endpoint URL.")
    parser.add_argument("--count", type=int, default=20, help="Number of measured requests.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests before measurement.")
    parser.add_argument("--timeout", type=float, default=3.0, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Respect HTTP(S)_PROXY and related environment variables.",
    )
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = measure_rtt(
        url=args.url,
        count=args.count,
        timeout=args.timeout,
        warmup=args.warmup,
        use_env_proxy=args.use_env_proxy,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_report(result))
    return 0 if result["success_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
