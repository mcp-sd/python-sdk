"""S2SPP protocol exception hierarchy."""

from __future__ import annotations

from typing import Optional


class S2SPError(Exception):
    """Base exception for all S2SP protocol errors."""

    def __init__(self, message: str, transfer_id: Optional[str] = None):
        self.transfer_id = transfer_id
        super().__init__(message)


class TransferDeniedError(S2SPError):
    """Raised when a transfer is denied by sender or receiver."""

    def __init__(
        self,
        message: str,
        transfer_id: Optional[str] = None,
        denied_by: str = "unknown",
        reason: Optional[str] = None,
    ):
        self.denied_by = denied_by
        self.reason = reason
        super().__init__(message, transfer_id)


class TransferFailedError(S2SPError):
    """Raised when a transfer fails during execution."""

    def __init__(
        self,
        message: str,
        transfer_id: Optional[str] = None,
        bytes_transferred: int = 0,
    ):
        self.bytes_transferred = bytes_transferred
        super().__init__(message, transfer_id)


class TransferTimeoutError(S2SPError):
    """Raised when a transfer times out."""

    pass


class InvalidTokenError(S2SPError):
    """Raised when a transfer token is invalid or expired."""

    pass


class InvalidStateTransitionError(S2SPError):
    """Raised when an invalid state transition is attempted."""

    def __init__(
        self,
        message: str,
        transfer_id: Optional[str] = None,
        from_state: Optional[str] = None,
        to_state: Optional[str] = None,
    ):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(message, transfer_id)
