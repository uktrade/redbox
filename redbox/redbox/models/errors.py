class AIError(Exception):
    """Basic error class for problems with the AI components."""


class QuestionLengthError(AIError):
    """Question length exceeds context window."""


class NoDocumentSelected(AIError):
    """No documents selected."""


class InactiveFileError(ValueError):
    def __init__(self, file):
        super().__init__(f"{file.pk} is inactive, status is {file.status}")
