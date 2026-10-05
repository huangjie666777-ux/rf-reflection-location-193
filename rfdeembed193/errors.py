class TouchstoneError(ValueError):
    def __init__(self, message: str, *, filename: str = "<upload>", line: int | None = None):
        self.filename = filename
        self.line = line
        location = filename if line is None else f"{filename}:{line}"
        super().__init__(f"{location}: {message}")


class ComputationError(ValueError):
    def __init__(self, message: str, *, frequency_hz: float | None = None):
        self.frequency_hz = frequency_hz
        if frequency_hz is None:
            super().__init__(message)
        else:
            super().__init__(f"{message} (frequency: {frequency_hz:g} Hz)")
