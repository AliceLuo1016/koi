#!/usr/bin/env python3
"""Generate realistic mock log lines for testing log analysis."""

import argparse
import random
import sys
from datetime import datetime, timedelta


SERVICES = ["api-gateway", "auth-service", "db-pool", "cache-redis", "worker-queue", "scheduler"]
INFO_MESSAGES = [
    "Request processed successfully in {ms}ms",
    "Health check passed",
    "Connection pool: {n} active, {m} idle",
    "Cache hit ratio: {pct}%",
    "Scheduled job completed: cleanup_old_sessions",
    "User login successful uid={uid}",
    "Metrics flushed: {n} datapoints",
]
WARN_MESSAGES = [
    "Slow query detected: {ms}ms (threshold 500ms)",
    "Connection pool nearing capacity: {n}/{m}",
    "Retry attempt {n}/3 for upstream call",
    "Memory usage at {pct}% of limit",
    "Rate limit approaching for client {uid}",
    "Certificate expires in {n} days",
]
ERROR_MESSAGES = [
    "Connection refused: upstream service unavailable",
    "Timeout after 30s waiting for db response",
    "Out of memory: cannot allocate {n}MB",
    "Unhandled exception in request handler: NullPointerError",
    "Disk usage critical: {pct}% used on /data",
    "Authentication failed: invalid token for uid={uid}",
    "Circuit breaker OPEN for auth-service after 5 failures",
]


def rand_sub(msg: str) -> str:
    """Fill placeholders with random values."""
    return (msg
            .replace("{ms}", str(random.randint(1, 5000)))
            .replace("{n}", str(random.randint(1, 100)))
            .replace("{m}", str(random.randint(50, 200)))
            .replace("{pct}", str(random.randint(50, 99)))
            .replace("{uid}", f"u{random.randint(1000,9999)}")
            )


def generate(count: int = 100, error_rate: float = 0.05) -> None:
    ts = datetime.now() - timedelta(minutes=count)
    for i in range(count):
        ts += timedelta(seconds=random.randint(1, 60))
        r = random.random()
        if r < error_rate:
            level, msg = "ERROR", random.choice(ERROR_MESSAGES)
        elif r < error_rate + 0.10:
            level, msg = "WARN", random.choice(WARN_MESSAGES)
        else:
            level, msg = "INFO", random.choice(INFO_MESSAGES)
        service = random.choice(SERVICES)
        print(f"{ts.strftime('%Y-%m-%d %H:%M:%S')} [{level}] [{service}] {rand_sub(msg)}")


def main():
    parser = argparse.ArgumentParser(description="Generate mock log lines")
    parser.add_argument("--count", type=int, default=100, help="Number of lines (default: 100)")
    parser.add_argument("--error-rate", type=float, default=0.05, help="Error probability 0-1 (default: 0.05)")
    args = parser.parse_args()
    generate(count=args.count, error_rate=args.error_rate)


if __name__ == "__main__":
    main()
