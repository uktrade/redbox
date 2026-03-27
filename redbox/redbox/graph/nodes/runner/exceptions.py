from __future__ import annotations

from typing import Any


class BaseToolRunnerException(Exception):
    """Base exception for all LLM tool runner errors."""

    def __init__(self, message: str, *, tool_name: str | None = None, **context: Any) -> None:
        self.tool_name = tool_name
        self.context = context
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        prefix = f"[{self.tool_name}] " if self.tool_name else ""
        return f"{prefix}{base}"


# Not Found


class ToolNotFoundError(BaseToolRunnerException):
    """Raised when a requested tool does not exist in the registry."""

    def __init__(self, tool_name: str, *, available_tools: list[str] | None = None) -> None:
        self.available_tools = available_tools or []
        hint = f" Available tools: {', '.join(self.available_tools)}." if self.available_tools else ""
        super().__init__(
            f"Tool '{tool_name}' not found.{hint}",
            tool_name=tool_name,
        )


# Validation


class ToolValidationError(BaseToolRunnerException):
    """Raised when tool input or output fails schema validation."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        field: str | None = None,
        value: Any = None,
    ) -> None:
        self.field = field
        self.value = value
        super().__init__(message, tool_name=tool_name, field=field, value=value)


class ToolInputValidationError(ToolValidationError):
    """Raised when the arguments passed to a tool are invalid."""

    def __init__(
        self,
        tool_name: str,
        *,
        field: str | None = None,
        value: Any = None,
        reason: str = "invalid input",
    ) -> None:
        field_info = f" (field: '{field}')" if field else ""
        super().__init__(
            f"Invalid input for tool '{tool_name}'{field_info}: {reason}.",
            tool_name=tool_name,
            field=field,
            value=value,
        )


class ToolOutputValidationError(ToolValidationError):
    """Raised when a tool's return value does not match the expected schema."""

    def __init__(
        self,
        tool_name: str,
        *,
        reason: str = "unexpected output format",
    ) -> None:
        super().__init__(
            f"Tool '{tool_name}' returned an invalid output: {reason}.",
            tool_name=tool_name,
        )


# Execution


class ToolExecutionError(BaseToolRunnerException):
    """Base for errors that occur during tool execution."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        cause: BaseException | None = None,
        **context: Any,
    ) -> None:
        self.cause = cause
        super().__init__(message, tool_name=tool_name, **context)
        if cause is not None:
            self.__cause__ = cause


class ToolTimeoutError(ToolExecutionError):
    """Raised when a tool exceeds its allowed execution time."""

    def __init__(self, tool_name: str, *, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout_seconds:.1f}s.",
            tool_name=tool_name,
            timeout_seconds=timeout_seconds,
        )


class ToolFailedError(ToolExecutionError):
    """Raised when a tool raises an unexpected error during execution."""

    def __init__(
        self,
        tool_name: str,
        *,
        cause: BaseException | None = None,
        exit_code: int | None = None,
        stderr: str | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.stderr = stderr
        detail = stderr or (str(cause) if cause else "unknown error")
        code_info = f" (exit code: {exit_code})" if exit_code is not None else ""
        super().__init__(
            f"Tool '{tool_name}' failed{code_info}: {detail}",
            tool_name=tool_name,
            cause=cause,
            exit_code=exit_code,
            stderr=stderr,
        )


class ToolRateLimitError(ToolExecutionError):
    """Raised when a tool is called more frequently than its rate limit allows."""

    def __init__(
        self,
        tool_name: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        hint = f" Retry after {retry_after_seconds:.1f}s." if retry_after_seconds is not None else ""
        super().__init__(
            f"Tool '{tool_name}' rate limit exceeded.{hint}",
            tool_name=tool_name,
            retry_after_seconds=retry_after_seconds,
        )


class ToolPermissionError(ToolExecutionError):
    """Raised when the caller lacks permission to invoke the requested tool."""

    def __init__(
        self,
        tool_name: str,
        *,
        required_permission: str | None = None,
    ) -> None:
        self.required_permission = required_permission
        detail = f" Required permission: '{required_permission}'." if required_permission else ""
        super().__init__(
            f"Permission denied for tool '{tool_name}'.{detail}",
            tool_name=tool_name,
            required_permission=required_permission,
        )


# Registry


class ToolRegistryError(BaseToolRunnerException):
    """Base for errors related to tool registration and loading."""


class ToolAlreadyRegisteredError(ToolRegistryError):
    """Raised when attempting to register a tool that is already present."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"Tool '{tool_name}' is already registered. Use replace=True to overwrite.",
            tool_name=tool_name,
        )


class ToolLoadError(ToolRegistryError):
    """Raised when a tool module or definition cannot be loaded."""

    def __init__(
        self,
        tool_name: str,
        *,
        cause: BaseException | None = None,
        path: str | None = None,
    ) -> None:
        self.path = path
        location = f" (path: '{path}')" if path else ""
        detail = str(cause) if cause else "unknown error"
        super().__init__(
            f"Failed to load tool '{tool_name}'{location}: {detail}",
            tool_name=tool_name,
            cause=cause,
            path=path,
        )
        if cause is not None:
            self.__cause__ = cause
