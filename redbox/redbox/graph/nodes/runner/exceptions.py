class BaseToolRunnerException(Exception):
    def __init__(self, message: str) -> None:
        super.__init__(message)


class ToolNotFoundError(BaseToolRunnerException):
    pass


class ToolValidationError(BaseToolRunnerException):
    pass


class ToolExecutionError(BaseToolRunnerException):
    pass


class ToolTimeoutError(BaseToolRunnerException):
    pass


class ToolRegistryError(BaseToolRunnerException):
    pass
