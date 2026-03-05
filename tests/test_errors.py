"""Tests for error classification and retry delay extraction."""

import pytest

from koi.errors import (
    CONTEXT_OVERFLOW_PATTERNS,
    KoiAPIError,
    KoiAuthError,
    KoiBillingError,
    KoiConnectionError,
    KoiContextOverflowError,
    KoiError,
    KoiOverloadedError,
    KoiRateLimitError,
    KoiServerError,
    classify_http_error,
    extract_retry_delay,
)


class TestClassifyHttpError:
    def test_401_is_auth_error(self):
        err = classify_http_error(401, "Unauthorized")
        assert isinstance(err, KoiAuthError)
        assert not err.retryable

    def test_403_is_auth_error(self):
        err = classify_http_error(403, "Forbidden")
        assert isinstance(err, KoiAuthError)
        assert not err.retryable

    def test_402_is_billing_error(self):
        err = classify_http_error(402, "Payment Required")
        assert isinstance(err, KoiBillingError)
        assert not err.retryable

    def test_billing_pattern_in_body(self):
        err = classify_http_error(400, '{"error": {"message": "insufficient funds"}}')
        assert isinstance(err, KoiBillingError)
        assert not err.retryable

    def test_quota_exceeded_is_billing(self):
        err = classify_http_error(400, "quota exceeded for this account")
        assert isinstance(err, KoiBillingError)

    def test_429_is_rate_limit(self):
        err = classify_http_error(429, "Too Many Requests")
        assert isinstance(err, KoiRateLimitError)
        assert err.retryable

    def test_rate_limit_in_body_non_429(self):
        err = classify_http_error(400, "rate_limit exceeded")
        assert isinstance(err, KoiRateLimitError)
        assert err.retryable

    def test_resource_exhausted_in_body(self):
        err = classify_http_error(400, "resource_exhausted: try again later")
        assert isinstance(err, KoiRateLimitError)
        assert err.retryable

    def test_overloaded_body_with_500(self):
        err = classify_http_error(500, '{"error": "overloaded"}')
        assert isinstance(err, KoiOverloadedError)
        assert err.retryable

    def test_529_is_overloaded(self):
        err = classify_http_error(529, "Service overloaded")
        assert isinstance(err, KoiOverloadedError)
        assert err.retryable

    def test_500_is_server_error(self):
        err = classify_http_error(500, "Internal Server Error")
        assert isinstance(err, KoiServerError)
        assert err.retryable

    def test_502_is_server_error(self):
        err = classify_http_error(502, "Bad Gateway")
        assert isinstance(err, KoiServerError)
        assert err.retryable

    def test_503_is_server_error(self):
        err = classify_http_error(503, "Service Unavailable")
        assert isinstance(err, KoiServerError)
        assert err.retryable

    def test_504_is_server_error(self):
        err = classify_http_error(504, "Gateway Timeout")
        assert isinstance(err, KoiServerError)
        assert err.retryable

    def test_context_overflow_anthropic(self):
        err = classify_http_error(400, "prompt is too long: 213462 tokens > 200000 maximum")
        assert isinstance(err, KoiContextOverflowError)
        assert not err.retryable

    def test_context_overflow_openai(self):
        err = classify_http_error(400, "Your input exceeds the context window of this model")
        assert isinstance(err, KoiContextOverflowError)

    def test_context_overflow_google(self):
        err = classify_http_error(
            400,
            "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)",
        )
        assert isinstance(err, KoiContextOverflowError)

    def test_unknown_400_is_generic(self):
        err = classify_http_error(400, "Bad Request: invalid JSON")
        assert isinstance(err, KoiAPIError)
        assert not isinstance(err, KoiAuthError)
        assert not err.retryable

    def test_retry_after_passed_through(self):
        err = classify_http_error(429, "Too Many Requests", retry_after=30.0)
        assert isinstance(err, KoiRateLimitError)
        assert err.retry_after == 30.0

    def test_service_unavailable_in_body(self):
        err = classify_http_error(400, "service unavailable, please retry")
        assert isinstance(err, KoiRateLimitError)
        assert err.retryable


class TestExtractRetryDelay:
    def test_retry_after_header(self):
        delay = extract_retry_delay("", headers={"retry-after": "30"})
        assert delay == 30.0

    def test_x_ratelimit_reset_after_header(self):
        delay = extract_retry_delay("", headers={"x-ratelimit-reset-after": "15.5"})
        assert delay == 15.5

    def test_reset_after_seconds_in_body(self):
        delay = extract_retry_delay("Your quota will reset after 39s")
        assert delay == 39.0

    def test_reset_after_hms_in_body(self):
        delay = extract_retry_delay("Your quota will reset after 1h30m10s")
        assert delay == 5410.0

    def test_retry_in_seconds(self):
        delay = extract_retry_delay("Please retry in 5s")
        assert delay == 5.0

    def test_retry_in_milliseconds(self):
        delay = extract_retry_delay("Please retry in 500ms")
        assert delay == 0.5

    def test_retry_delay_json_field(self):
        delay = extract_retry_delay('{"retryDelay": "34.07s"}')
        assert abs(delay - 34.07) < 0.01

    def test_no_delay_found(self):
        delay = extract_retry_delay("Some random error")
        assert delay is None

    def test_headers_none(self):
        delay = extract_retry_delay("retry in 10s", headers=None)
        assert delay == 10.0

    def test_header_takes_priority(self):
        delay = extract_retry_delay("retry in 10s", headers={"retry-after": "5"})
        assert delay == 5.0  # Header wins


class TestContextOverflowPatterns:
    """Test that overflow patterns match real provider error messages."""

    @pytest.mark.parametrize(
        "msg",
        [
            "prompt is too long: 213462 tokens > 200000 maximum",
            "input is too long for requested model",
            "Your input exceeds the context window of this model",
            "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)",
            "This model's maximum prompt length is 131072",
            "Please reduce the length of the messages or completion",
            "This endpoint's maximum context length is 8192 tokens",
            "prompt token count of 50000 exceeds the limit of 32000",
            "the request exceeds the available context size",
            "tokens to keep from the initial prompt is greater than the context length",
            "context window exceeds limit",
            "Your request exceeded model token limit",
            "context_length_exceeded",
            "too many tokens in the request",
            "token limit exceeded",
        ],
    )
    def test_overflow_pattern_matches(self, msg):
        assert CONTEXT_OVERFLOW_PATTERNS.search(msg), f"Pattern should match: {msg}"


class TestKoiErrorHierarchy:
    def test_all_api_errors_inherit_from_koi_error(self):
        for cls in [
            KoiAPIError,
            KoiRateLimitError,
            KoiAuthError,
            KoiBillingError,
            KoiOverloadedError,
            KoiServerError,
            KoiContextOverflowError,
        ]:
            assert issubclass(cls, KoiAPIError)
            assert issubclass(cls, KoiError)

    def test_connection_error_inherits_from_koi_error(self):
        err = KoiConnectionError("timeout")
        assert isinstance(err, KoiError)
        assert err.retryable

    def test_api_error_attributes(self):
        err = KoiAPIError("test", status_code=418, error_text="teapot", retryable=False)
        assert err.status_code == 418
        assert err.error_text == "teapot"
        assert not err.retryable
        assert str(err) == "test"
