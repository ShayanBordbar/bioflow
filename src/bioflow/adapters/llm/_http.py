"""HTTP plumbing shared by the provider clients.

Both clients POST JSON, retry the same statuses with the same backoff, and must never
let a raw ``httpx`` exception escape. Keeping that in one place means the two providers
cannot drift apart on retry semantics -- which would be an invisible difference until the
day one of them hammers a rate-limited endpoint.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
)

from bioflow.core.models import BioflowError, ErrorCode, JSONValue, StructuredError

if TYPE_CHECKING:
    from tenacity.wait import WaitBaseT

DEFAULT_MAX_TOKENS = 16_000
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_ATTEMPTS = 3
MAX_RETRY_AFTER_SECONDS = 60.0

# 5xx is handled separately and covers 529 overloaded_error, the common one.
RETRYABLE_STATUSES = frozenset({408, 409, 429})


class RetryableStatus(Exception):
    """Internal marker for an HTTP status worth another attempt."""

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


def is_retryable(exc: BaseException) -> bool:
    """Whether a raised exception should trigger another attempt."""
    return isinstance(exc, RetryableStatus | httpx.TransportError)


def should_retry_status(status_code: int) -> bool:
    """Whether an HTTP status is worth another attempt."""
    return status_code in RETRYABLE_STATUSES or status_code >= 500


def honour_retry_after(state: RetryCallState) -> None:
    """Sleep for the server-provided delay when a 429 supplies one."""
    outcome = state.outcome
    if outcome is None or not outcome.failed:
        return
    exc = outcome.exception()
    if not isinstance(exc, RetryableStatus):
        return
    header = exc.response.headers.get("retry-after")
    if header is None:
        return
    try:
        delay = float(header)
    except ValueError:
        return
    time.sleep(min(delay, MAX_RETRY_AFTER_SECONDS))


def http_error(
    response: httpx.Response, *, raised_by: str, exhausted: bool = False
) -> BioflowError:
    """Turn an error response into a StructuredError carrying the API's own explanation."""
    try:
        body = response.json()
        api_error = body.get("error", {}) if isinstance(body, dict) else {}
        api_type = str(api_error.get("type", ""))
        api_message = str(api_error.get("message", ""))
    except ValueError:
        api_type, api_message = "", response.text[:500]

    status = response.status_code
    if status == 429:
        code, remediation = ErrorCode.RATE_LIMITED, "Back off and retry."
    elif status in (401, 403):
        code, remediation = ErrorCode.VALIDATION_FAILED, "Check your credentials."
    elif status >= 500:
        code, remediation = ErrorCode.EXTERNAL_TOOL_FAILED, "The server is unhealthy; retry later."
    else:
        code, remediation = ErrorCode.VALIDATION_FAILED, "Fix the request and try again."

    prefix = "gave up after retries: " if exhausted else ""
    detail = api_message or api_type or "no detail"
    return BioflowError(
        StructuredError(
            error_code=code,
            message=f"{prefix}{status} from the model API: {detail}",
            remediation=remediation,
            details={"status": status, "type": api_type},
            retryable=status == 429 or status >= 500,
            raised_by=raised_by,
        )
    )


def post_json(
    *,
    client: httpx.Client,
    url: str,
    body: dict[str, JSONValue],
    headers: dict[str, str],
    max_attempts: int,
    retry_wait: WaitBaseT,
    raised_by: str,
    endpoint_label: str,
) -> dict[str, Any]:
    """POST a JSON body with retries, returning the decoded response.

    Every failure leaves as a :class:`BioflowError`; no ``httpx`` exception escapes.
    """
    try:
        for attempt in Retrying(
            retry=retry_if_exception(is_retryable),
            stop=stop_after_attempt(max_attempts),
            wait=retry_wait,
            before_sleep=honour_retry_after,
            reraise=True,
        ):
            with attempt:
                response = client.post(url, json=body, headers=headers)
                if should_retry_status(response.status_code):
                    raise RetryableStatus(response)
                if response.status_code >= 400:
                    raise http_error(response, raised_by=raised_by)
                decoded: dict[str, Any] = response.json()
                return decoded
    except RetryableStatus as exc:
        raise http_error(exc.response, raised_by=raised_by, exhausted=True) from exc
    except httpx.TransportError as exc:
        raise BioflowError(
            StructuredError(
                error_code=ErrorCode.EXTERNAL_TOOL_FAILED,
                message=f"could not reach {endpoint_label}: {exc}",
                remediation="Check connectivity and the configured endpoint, then retry.",
                details={"endpoint": endpoint_label},
                retryable=True,
                raised_by=raised_by,
            )
        ) from exc
    raise AssertionError("unreachable")  # pragma: no cover
