"""What a client is told when something fails. Raw exception text can carry
upstream request ids, URLs and internal detail, so it goes to the log and the
client gets a fixed message per kind of failure."""
from __future__ import annotations

from anthropic import AnthropicError, APITimeoutError, AuthenticationError, RateLimitError


def user_facing_error(exc: BaseException) -> str:
    if isinstance(exc, RateLimitError):
        return "The AI service is busy right now. Try again in a few minutes."
    if isinstance(exc, APITimeoutError):
        return "The AI service took too long to respond."
    if isinstance(exc, AuthenticationError):
        return "The server's AI service key was rejected."
    if isinstance(exc, AnthropicError):
        return "The AI service returned an error."
    return "Something went wrong during the analysis."
