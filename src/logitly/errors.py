"""Exceptions raised by logitly."""


class LogitlyError(Exception):
    """Base class for every logitly error."""


class BackendError(LogitlyError):
    """The inference backend could not be created or executed."""


class TokenizationError(LogitlyError):
    """Option labels cannot be mapped onto distinct token ids."""


class QuestionError(LogitlyError):
    """A question was declared with an invalid set of options."""


class CalibrationError(LogitlyError):
    """Calibration data or parameters are invalid."""
