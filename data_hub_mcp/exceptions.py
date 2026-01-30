class ToolInputError(ValueError):
    """
    Raised when sector-related tools encounter invalid input.
    For example, neither sector_name nor company_name is provided.
    """

    def __init__(self, required_inputs: list[str] | None = None, exclusive_inputs: list[str] | None = None):
        msg_info = ""
        if len(required_inputs or []) > 0:
            msg_info = f"Missing required fields: {', '.join(required_inputs)}"
        if len(exclusive_inputs or []) > 0:
            msg_info = f"Must provide one of the following fields: {', '.join(exclusive_inputs)}"
        message = f"Invalid input for tool. {msg_info}"
        super().__init__(message)


class NoCompanyOrSectorError(ToolInputError):
    def __init__(self):
        super().__init__(exclusive_inputs=["sector_name", "company_name"])
