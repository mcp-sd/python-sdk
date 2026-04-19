"""DDI buffer — in-process body store for sync-mode S2SP.

In sync mode the resource server returns abstract + body rows inline in a
single MCP tool response. The dispatcher extracts the body into a
:class:`DDIBuffer` and replaces it with an opaque ``ddi://`` handle before
the response reaches the LLM context. When the agent later calls a consumer
tool and passes the handle, the dispatcher resolves it back into inlined
body rows.

The buffer is session-scoped and holds body rows keyed by cryptographically
random handles. Handles are single-use by default, matching S2SP's async
presigned-URL semantics.
"""

from __future__ import annotations

import secrets
from typing import Dict, List


class DDIBuffer:
    """In-memory store for body rows awaiting consumer delivery."""

    SCHEME = "ddi://"

    def __init__(self) -> None:
        self._store: Dict[str, List[dict]] = {}

    def put(self, rows: List[dict]) -> str:
        """Stash body rows, return a fresh ``ddi://`` handle."""
        handle = f"{self.SCHEME}{secrets.token_urlsafe(16)}"
        self._store[handle] = rows
        return handle

    def take(self, handle: str) -> List[dict]:
        """Fetch rows by handle and remove (single-use)."""
        return self._store.pop(handle, [])

    def peek(self, handle: str) -> List[dict]:
        """Fetch rows by handle without removing."""
        return self._store.get(handle, [])

    @classmethod
    def is_handle(cls, value: object) -> bool:
        return isinstance(value, str) and value.startswith(cls.SCHEME)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, handle: str) -> bool:
        return handle in self._store
