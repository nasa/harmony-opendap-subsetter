""" This module contains custom exceptions specific to the Harmony OPeNDAP
    SubSetter (HOSS). These exceptions are intended to allow for easier
    debugging of the expected errors that may occur during an invocation of the
    HOSS.

"""


class CustomError(Exception):
    """Base class for exceptions in HOSS. This base class allows for future
    work, such as assigning exit codes for specific failure modes.

    """

    def __init__(self, exception_type, message):
        self.exception_type = exception_type
        self.message = message
        super().__init__(self.message)


class InvalidInputGeoJSON(CustomError):
    """This exception is raised when a supplied GeoJSON object does not
    adhere the GeoJSON schema. For example, if a GeoJSON geometry does not
    contain either a `bbox` or a `coordinates` attribute.

    """

    def __init__(self):
        super().__init__(
            'InvalidInputGeoJSON',
            'The supplied shape file cannot be parsed according '
            'to the GeoJSON format defined in RFC 7946.',
        )


class InvalidNamedDimension(CustomError):
    """This exception is raised when a user-supplied dimension name
    is not in the list of required dimensions for the subset.

    """

    def __init__(self, dimension_name):
        super().__init__(
            'InvalidNamedDimension',
            f'"{dimension_name}" is not a dimension for '
            'any of the requested variables.',
        )


class InvalidRequestedRange(CustomError):
    """This exception is raised when a user-supplied dimension range lies
    entirely outside the range of a dimension with an associated bounds
    variable.

    """

    def __init__(self):
        super().__init__(
            'InvalidRequestedRange',
            'Input request specified range outside supported ' 'dimension range',
        )


class MissingGridMappingMetadata(CustomError):
    """This exception is raised when HOSS tries to obtain the `grid_mapping`
    metadata attribute for a projected variable and it is not present in
    either the input granule or the CF-Convention overrides defined in the
    earthdata-varinfo configuration file.

    """

    def __init__(self, variable_name):
        super().__init__(
            'MissingGridMappingMetadata',
            f'Projected variable "{variable_name}" does not have '
            'an associated "grid_mapping" metadata attribute.',
        )


class MissingGridMappingVariable(CustomError):
    """This exception is raised when HOSS tries to extract attributes from a
    `grid_mapping` variable referred to by another variable, but that
    `grid_mapping` variable is not present in the `.dmr` for that granule.

    """

    def __init__(self, grid_mapping_variable, referring_variable):
        super().__init__(
            'MissingGridMappingVariable',
            f'Grid mapping variable "{grid_mapping_variable}" '
            f'referred to by variable "{referring_variable}" is '
            'not present in granule .dmr file.',
        )


class MissingSpatialSubsetInformation(CustomError):
    """This exception is raised when HOSS reaches a branch of the code that
    requires spatial subset information, but neither a bounding box, nor a
    shape file is specified.

    """

    def __init__(self):
        super().__init__(
            'MissingSpatialSubsetInformation',
            'Either a bounding box or shape file must be '
            'specified when performing spatial subsetting.',
        )


class UnsupportedShapeFileFormat(CustomError):
    """This exception is raised when the shape file included in the input
    Harmony message is not GeoJSON.

    """

    def __init__(self, shape_file_mime_type: str):
        super().__init__(
            'UnsupportedShapeFileFormat',
            f'Shape file format "{shape_file_mime_type}" not ' 'supported.',
        )


class UnsupportedTemporalUnits(CustomError):
    """This exception is raised when the 'units' metadata attribute contains
    a temporal unit that is not supported by HOSS.

    """

    def __init__(self, units_string):
        super().__init__(
            'UnsupportedTemporalUnits',
            f'Temporal units "{units_string}" not supported.',
        )


class UrlAccessFailed(CustomError):
    """This exception is raised when an HTTP request for a given URL has a non
    500 error, and is therefore not retried.

    """

    def __init__(self, url, status_code):
        super().__init__('UrlAccessFailed', f'{status_code} error retrieving: {url}')
