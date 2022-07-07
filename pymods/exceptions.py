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


class InvalidInputGeoJSON(CustomError):
    """ This exception is raised when a supplied GeoJSON object does not
        adhere the GeoJSON schema. For example, if a GeoJSON geometry does not
        contain either a `bbox` or a `coordinates` attribute.

    """
    def __init__(self):
        super().__init__('InvalidInputGeoJSON',
                         'The supplied shape file cannot be parsed according '
                         'to the GeoJSON format defined in RFC 7946.')


class UnsupportedShapeFileFormat(CustomError):
    """ This exception is raised when the shape file included in the input
        Harmony message is not GeoJSON.

    """
    def __init__(self, shape_file_mime_type: str):
        super().__init__('UnsupportedShapeFileFormat',
                         f'Shape file format "{shape_file_mime_type}" not '
                         'supported.')


class UnsupportedTemporalUnits(CustomError):
    """ This exception is raised when the 'units' metadata attribute contains
        a temporal unit that is not supported by HOSS.

    """
    def __init__(self, units_string):
        super().__init__('UnsupportedTemporalUnits',
                         f'Temporal units "{units_string}" not supported.')


class UrlAccessFailed(CustomError):
    """ This exception is raised when an HTTP request for a given URL has a non
        500 error, and is therefore not retried.

    """
    def __init__(self, url, status_code):
        super().__init__('UrlAccessFailed',
                         f'{status_code} error retrieving: {url}')


class UrlAccessFailedWithRetries(CustomError):
    """ This exception is raised when an HTTP request for a given URL has
        failed a specified number of times.

    """
    def __init__(self, url):
        super().__init__('UrlAccessFailedWithRetries',
                         f'URL: {url} was unsuccessfully requested the '
                         'maximum number of times.')


class InvalidNamedDimension(CustomError):
    """ This exception is raised when a user-supplied dimension name
        is not in the list of required dimensions for the subset.

    """
    def __init__(self, dim_name):
        super().__init__('InvalidNamedDimension',
                         f'"{dim_name}" is not a dimension for '
                         'any of the requested variables.')
