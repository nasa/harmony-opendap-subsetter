""" This module contains custom exceptions specific to the Harmonized Variable
    Subsetter service. These exceptions are intended to allow for easier
    debugging of the expected errors that may occur during an invocation of the
    variable subsetter.

"""


class CustomError(Exception):
    """ Base class for exceptions in the variable subsetter. This base class
        allows for future work, such as assigning exit codes for specific
        failure modes.

    """
    def __init__(self, exception_type, message):
        self.exception_type = exception_type
        self.message = message
        super().__init__(self.message)


class PydapRetrievalError(CustomError):
    """ This exception is raised when pydap fails to retrieve a dataset from
        a specified URL.

    """
    def __init__(self, message):
        super().__init__('PydapRetrievalError', message)

class URLResponseError(CustomError):
    """ This exception is raised when response from a specified URL fails."""

    def __init__(self, message):
        super().__init__('DmrRetrievalError', message)
