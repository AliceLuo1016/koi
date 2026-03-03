"""Koi error types for API error classification."""

import json
import re
from typing import Optional


class KoiError(Exception):
    """Base error for all Koi errors."""
    pass


class KoiAPIError(KoiError):
    """Base for API errors. Stores status_code, error_text, retryable flag."""
    def __init__(self, message: str, status_code: int = 0, error_text: str = "", retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.error_text = error_text
        self.retryable = retryable


class KoiRateLimitError(KoiAPIError):
    """429 or rate limit detected in body."""
    def __init__(self, message: str, status_code: int = 429, error_text: str = "", retry_after: Optional[float] = None):
        super().__init__(message, status_code, error_text, retryable=True)
        self.retry_after = retry_after


class KoiAuthError(KoiAPIError):
    """401, 403 — not retryable."""
    def __init__(self, message: str, status_code: int = 401, error_text: str = ""):
        super().__init__(message, status_code, error_text, retryable=False)


class KoiBillingError(KoiAPIError):
    """402 or payment-related — not retryable."""
    def __init__(self, message: str, status_code: int = 402, error_text: str = ""):
        super().__init__(message, status_code, error_text, retryable=False)


class KoiOverloadedError(KoiAPIError):
    """529 or 'overloaded' in body — retryable."""
    def __init__(self, message: str, status_code: int = 529, error_text: str = ""):
        super().__init__(message, status_code, error_text, retryable=True)


class KoiServerError(KoiAPIError):
    """500, 502, 503, 504 — retryable."""
    def __init__(self, message: str, status_code: int = 500, error_text: str = ""):
        super().__init__(message, status_code, error_text, retryable=True)


class KoiContextOverflowError(KoiAPIError):
    """Context too long — not retryable (needs compaction)."""
    def __init__(self, message: str, status_code: int = 400, error_text: str = ""):
        super().__init__(message, status_code, error_text, retryable=False)


class KoiConnectionError(KoiError):
    """Connection/timeout — retryable."""
    def __init__(self, message: str):
        super().__init__(message)
        self.retryable = True


# --- Classification helpers ---

# Body text patterns indicating retryable errors (from pi-ai)
_RETRYABLE_BODY_PATTERNS = re.compile(
    r"resource.?exhausted|rate.?limit|overloaded|service.?unavailable|other.?side.?closed|"
    r"too many requests|capacity|throttl",
    re.IGNORECASE,
)

# Body text patterns indicating billing/payment issues
_BILLING_PATTERNS = re.compile(
    r"insufficient.?funds|payment.?required|billing|quota.?exceeded|account.?suspended|"
    r"credit|deactivated|exceeded.*budget",
    re.IGNORECASE,
)

# Context overflow patterns (from pi-ai's overflow.js — 15+ providers)
CONTEXT_OVERFLOW_PATTERNS = re.compile(
    r"prompt is too long|"
    r"input is too long for requested model|"
    r"exceeds the context window|"
    r"input token count.*exceeds the maximum|"
    r"maximum prompt length is \d+|"
    r"reduce the length of the messages|"
    r"maximum context length is \d+ tokens|"
    r"exceeds the limit of \d+|"
    r"exceeds the available context size|"
    r"greater than the context length|"
    r"context window exceeds limit|"
    r"exceeded model token limit|"
    r"context.?length.?exceeded|"
    r"too many tokens|"
    r"token limit exceeded",
    re.IGNORECASE,
)


def classify_http_error(status_code: int, error_text: str, retry_after: Optional[float] = None) -> KoiAPIError:
    """Classify an HTTP error into a typed KoiAPIError."""

    # Check body text for context overflow first (can come as 400)
    if CONTEXT_OVERFLOW_PATTERNS.search(error_text):
        return KoiContextOverflowError(
            f"Context too long: {_extract_message(error_text)}",
            status_code=status_code, error_text=error_text,
        )

    # Status-code based classification
    if status_code == 401 or status_code == 403:
        return KoiAuthError(
            f"Authentication failed (HTTP {status_code})",
            status_code=status_code, error_text=error_text,
        )

    if status_code == 402 or _BILLING_PATTERNS.search(error_text):
        return KoiBillingError(
            f"Billing issue (HTTP {status_code}): {_extract_message(error_text)}",
            status_code=status_code, error_text=error_text,
        )

    if status_code == 429 or (status_code < 500 and _RETRYABLE_BODY_PATTERNS.search(error_text)):
        return KoiRateLimitError(
            f"Rate limited (HTTP {status_code})",
            status_code=status_code, error_text=error_text,
            retry_after=retry_after,
        )

    if status_code == 529 or (status_code >= 500 and "overloaded" in error_text.lower()):
        return KoiOverloadedError(
            f"Service overloaded (HTTP {status_code})",
            status_code=status_code, error_text=error_text,
        )

    if status_code in (500, 502, 503, 504):
        return KoiServerError(
            f"Server error (HTTP {status_code})",
            status_code=status_code, error_text=error_text,
        )

    # Default: non-retryable API error
    return KoiAPIError(
        f"API error (HTTP {status_code}): {_extract_message(error_text)}",
        status_code=status_code, error_text=error_text, retryable=False,
    )


def extract_retry_delay(error_text: str, headers: dict = None) -> Optional[float]:
    """Extract retry delay in seconds from headers and/or error body.

    Checks (in order):
    1. retry-after header (seconds or HTTP date)
    2. x-ratelimit-reset-after header
    3. Body patterns: "reset after Xs", "retry in Xs", "retryDelay": "Xs"

    Returns delay in seconds, or None.
    """
    if headers:
        # retry-after header
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass  # Could be HTTP date, skip for now

        # x-ratelimit-reset-after header (seconds)
        reset_after = headers.get("x-ratelimit-reset-after")
        if reset_after:
            try:
                return float(reset_after)
            except ValueError:
                pass

    # Body: "reset after 39s" or "reset after 18h31m10s"
    m = re.search(r"reset after (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", error_text, re.IGNORECASE)
    if m:
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        seconds = float(m.group(3))
        return (hours * 3600) + (minutes * 60) + seconds

    # Body: "retry in Xs" or "retry in Xms"
    m = re.search(r"retry in ([0-9.]+)(ms|s)", error_text, re.IGNORECASE)
    if m:
        value = float(m.group(1))
        if m.group(2).lower() == "ms":
            return value / 1000
        return value

    # Body: "retryDelay": "34.074s"
    m = re.search(r'"retryDelay":\s*"([0-9.]+)(ms|s)"', error_text, re.IGNORECASE)
    if m:
        value = float(m.group(1))
        if m.group(2).lower() == "ms":
            return value / 1000
        return value

    return None


def _extract_message(error_text: str) -> str:
    """Try to extract a clean error message from JSON error response."""
    try:
        parsed = json.loads(error_text)
        if isinstance(parsed, dict):
            # OpenAI/Anthropic style: {"error": {"message": "..."}}
            err = parsed.get("error", {})
            if isinstance(err, dict) and err.get("message"):
                return err["message"]
            # Simple: {"message": "..."}
            if parsed.get("message"):
                return parsed["message"]
    except (json.JSONDecodeError, TypeError):
        pass
    # Truncate raw text
    if len(error_text) > 200:
        return error_text[:200] + "..."
    return error_text
