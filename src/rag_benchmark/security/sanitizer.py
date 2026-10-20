"""Centralized secret sanitization and redaction layer."""

from __future__ import annotations

import copy
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED_STR = "[REDACTED]"

# Exact normalized field names representing credentials or secrets
EXACT_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "client_secret",
    "credential",
    "credentials",
    "private_key",
    "authorization",
    "bearer_token",
    "auth_token",
    "token",
}

# Suffixes and prefixes for compound field names (e.g. openai_api_key, secret_val)
SENSITIVE_SUFFIXES = (
    "_api_key",
    "_apikey",
    "_access_token",
    "_password",
    "_secret",
    "_private_key",
    "_token",
    "_credential",
    "_credentials",
    "_bearer_token",
)

SENSITIVE_PREFIXES = (
    "secret_",
    "api_key_",
    "apikey_",
    "access_token_",
    "bearer_token_",
    "private_key_",
)

# Explicitly allowed field names that must never be redacted despite containing substrings
EXPLICIT_SAFE_KEYS = {
    "author",
    "authors",
    "tokenizer",
    "tokenizer_name",
    "authorization_mode",
    "tokens",
    "token_count",
    "input_tokens",
    "output_tokens",
    "max_tokens",
    "total_tokens",
    "embedding_tokens",
}


def is_sensitive_key(key: str) -> bool:
    """Determine whether a dictionary key represents a credential or secret."""
    norm = key.strip().lower().replace("-", "_")
    if norm in EXPLICIT_SAFE_KEYS:
        return False
    if norm in EXACT_SENSITIVE_KEYS:
        return True
    if norm.endswith(SENSITIVE_SUFFIXES) or norm.startswith(SENSITIVE_PREFIXES):
        return True
    return False


def sanitize_url(url: str) -> str:
    """Scrub embedded credentials (userinfo and sensitive query params) from URLs."""
    if not isinstance(url, str) or "://" not in url:
        return url

    try:
        parts = urlsplit(url)
    except Exception:
        return url

    netloc = parts.netloc
    # Check for userinfo: user:pass@host or token@host
    if "@" in netloc:
        userinfo, host = netloc.split("@", 1)
        if ":" in userinfo:
            user = userinfo.split(":", 1)[0]
            sanitized_userinfo = f"{user}:{REDACTED_STR}"
        else:
            sanitized_userinfo = REDACTED_STR
        netloc = f"{sanitized_userinfo}@{host}"

    query = parts.query
    if query:
        q_params = parse_qsl(query, keep_blank_values=True)
        sanitized_params: list[tuple[str, str]] = []
        for k, v in q_params:
            if is_sensitive_key(k) or k.lower() in {"key", "token", "auth"}:
                sanitized_params.append((k, REDACTED_STR))
            else:
                sanitized_params.append((k, v))
        query = urlencode(sanitized_params)

    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


_BEARER_PATTERN = re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9_\-\.]{15,})\b")
_SK_PATTERN = re.compile(r"\b(sk-[A-Za-z0-9_\-]{15,})\b")


def sanitize_error_message(message: str) -> str:
    """Sanitize error messages to avoid accidental leakage of tokens or URL userinfo."""
    if not isinstance(message, str):
        return str(message)

    sanitized = _BEARER_PATTERN.sub(r"\1" + REDACTED_STR, message)
    sanitized = _SK_PATTERN.sub(REDACTED_STR, sanitized)

    # If message contains URLs with credentials, sanitize them
    if "://" in sanitized:
        words = sanitized.split()
        sanitized_words = [sanitize_url(w) if ("://" in w) else w for w in words]
        sanitized = " ".join(sanitized_words)

    return sanitized


def sanitize_dict_for_manifest(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact sensitive credential keys in structured dictionaries.

    Replaces credential values with '[REDACTED]' rather than deleting the key,
    ensuring transparency in reproducible manifests while protecting secrets.
    """
    clean: dict[str, Any] = {}
    for k, v in data.items():
        if is_sensitive_key(k):
            clean[k] = REDACTED_STR
        elif isinstance(v, dict):
            clean[k] = sanitize_dict_for_manifest(v)
        elif isinstance(v, list):
            clean[k] = [
                sanitize_dict_for_manifest(item) if isinstance(item, dict) else item for item in v
            ]
        elif isinstance(v, str) and ("://" in v):
            clean[k] = sanitize_url(v)
        else:
            clean[k] = copy.deepcopy(v)
    return clean
