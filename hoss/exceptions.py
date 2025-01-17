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
            'Input request specified range outside supported dimension range',
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


class MissingVariable(CustomError):
    """This exception is raised when HOSS tries to get variables and
    they are missing or empty.

    """

    def __init__(self, referring_variable):
        super().__init__(
            'MissingVariable',
            f'"{referring_variable}" is ' 'not present in source granule file.',
        )


class MissingCoordinateVariable(CustomError):
    """This exception is raised when HOSS tries to get latitude and longitude
    variables and they are missing or empty. These variables are referred to
    in the science variables with coordinate attributes.

    """

    def __init__(self, referring_variable):
        super().__init__(
            'MissingCoordinateVariable',
            f'Coordinate: "{referring_variable}" is '
            'not present in source granule file.',
        )


class InvalidIndexSubsetRequest(CustomError):
    """This exception is raised when HOSS tries to get dimensions or
    coordinate variables as part of a prefetch from opendap when there is
    a spatial or temporal request, and there are no prefetch variables
    returned.

    """

    def __init__(self, custom_msg):
        super().__init__(
            'InvalidIndexSubsetRequest',
            custom_msg,
        )


class InvalidCoordinateVariable(CustomError):
    """This exception is raised when HOSS tries to get latitude and longitude
    variables and they have fill values to the extent that it cannot be used.
    These variables are referred in the science variables with coordinate attributes.

    """

    def __init__(self, referring_variable):
        super().__init__(
            'InvalidCoordinateVariable',
            f'Coordinate: "{referring_variable}" is '
            'not valid in source granule file.',
        )


class IncompatibleCoordinateVariables(CustomError):
    """This exception is raised when HOSS tries to get latitude and longitude
    coordinate variable and they do not match in shape or have a size of 0.

    """

    def __init__(self, longitude_shape, latitude_shape):
        super().__init__(
            'IncompatibleCoordinateVariables',
            f'Longitude coordinate shape: "{longitude_shape}"'
            f'does not match the latitude coordinate shape: "{latitude_shape}"',
        )


class InvalidCoordinateData(CustomError):
    """This exception is raised when the data does not contain at least.
    two valid points. This could occur when there are too many fill values and distinct valid
    indices could not be obtained

    """

    def __init__(self, custom_msg):
        super().__init__(
            'InvalidCoordinateData',
            f'{custom_msg}',
        )


class InvalidCoordinateDataset(CustomError):
    """This exception is raised when there are too
    many fill values and two distinct valid indices
    could not be obtained

    """

    def __init__(self, coordinate_name):
        super().__init__(
            'InvalidCoordinateDataset',
            f'Cannot get valid indices for {coordinate_name}',
        )


class UnsupportedDimensionOrder(CustomError):
    """This exception is raised when the granule file included in the input
    request is not the nominal dimension order which is 'y,x'.

    """

    def __init__(self, dimension_order: str):
        super().__init__(
            'UnsupportedDimensionOrder',
            f'Dimension Order "{dimension_order}" not ' 'supported.',
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
