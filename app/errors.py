"""Application-specific exceptions for controller and logic layers."""

from __future__ import annotations


class LogicError(Exception):
    """Represent an expected business-logic error with an HTTP status code."""

    def __init__(self, message: str, status_code: int) -> None:
        """Store the error message and status code for controller translation."""

        super().__init__(message)
        self.message = message
        self.status_code = status_code
