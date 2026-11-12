from __future__ import annotations

import re


def redact_error(text: str) -> str:
    text = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", text, flags=re.I)
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text[:500]
