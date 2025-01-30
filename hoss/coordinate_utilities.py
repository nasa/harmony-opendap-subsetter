""" This module contains utility functions used for
    coordinate variables and functions to convert the
    coordinate variable data to projected x/y dimension values
"""

import numpy as np
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from varinfo import VariableFromDmr, VarInfoFromDmr

from hoss.exceptions import (
    IncompatibleCoordinateVariables,
    InvalidCoordinateData,
    InvalidCoordinateDataset,
    InvalidDimensionNames,
    MissingCoordinateVariable,
    MissingVariable,
    UnsupportedDimensionOrder,
)


def get_coordinate_variables(
    varinfo: VarInfoFromDmr,
    requested_variables: list[str],
) -> tuple[list[str], list[str]]:
    """This function returns latitude and longitude variable names from
    latitude and longitude variables listed in the CF-Convention coordinates
    metadata attribute. It returns them in a specific
    order [latitude_name, longitude_name]"
    """

    coordinate_variables = varinfo.get_references_for_attribute(
        requested_variables, 'coordinates'
    )

    latitude_coordinate_variables = [
        coordinate
        for coordinate in coordinate_variables
        if varinfo.get_variable(coordinate) is not None
        and varinfo.get_variable(coordinate).is_latitude()
    ]

    longitude_coordinate_variables = [
        coordinate
        for coordinate in coordinate_variables
        if varinfo.get_variable(coordinate) is not None
        and varinfo.get_variable(coordinate).is_longitude()
    ]

    return latitude_coordinate_variables, longitude_coordinate_variables


def get_variables_with_anonymous_dims(
    varinfo: VarInfoFromDmr, variables: set[str]
) -> set[str]:
    """
    returns a set of variables without any dimensions
    associated with it
    """

    return set(
        variable
        for variable in variables
        if (
            varinfo.get_variable(variable)
            and (
                (len(varinfo.get_variable(variable).dimensions) == 0)
                or (any_absent_dimension_variables(varinfo, variable))
            )
        )
    )


def any_absent_dimension_variables(varinfo: VarInfoFromDmr, variable: str) -> bool:
    """returns variable with fake dimensions - dimensions
    that have been created by opendap, but are not really
    dimension variables
    """

    return any(
        varinfo.get_variable(dimension) is None
        for dimension in varinfo.get_variable(variable).dimensions
    )


def get_dimension_array_names(
    varinfo: VarInfoFromDmr,
    variable_name: str,
) -> list[str]:
    """
    Returns the dimensions names from coordinate variables or from
    configuration
    """
    variable = varinfo.get_variable(variable_name)
    if variable is None:
        return []

    dimension_names = variable.dimensions

    if len(dimension_names) >= 2:
        # if the dimension is not a variable, was added as a
        # configuration entry,
        return dimension_names

    # getting dimension names from coordinates
    latitude_coordinates, longitude_coordinates = get_coordinate_variables(
        varinfo, [variable_name]
    )

    # for one variable, the coordinate array length will always be 1 or 0
    if len(latitude_coordinates) == 1 and len(longitude_coordinates) == 1:
        dimension_array_names = get_dimension_array_names_from_coordinates(
            varinfo, latitude_coordinates[0]
        )
    # if variable is a coordinate
    elif variable.is_latitude() or variable.is_longitude():
        dimension_array_names = get_dimension_array_names_from_coordinates(
            varinfo, variable_name
        )
    else:
        dimension_array_names = []

    return dimension_array_names


def get_dimension_array_names_from_coordinates(
    varinfo: VarInfoFromDmr, variable_name: str
) -> str:
    """returns the x-y variable names that would
    match the group of the input variable. The 'dim_y' dimension
    and 'dim_x' names are returned with the group pathname

    """
    variable = varinfo.get_variable(variable_name)

    if variable is not None:
        dimension_array_names = [
            f'{variable.group_path}/dim_y',
            f'{variable.group_path}/dim_x',
        ]
    else:
        raise MissingVariable(variable_name)

    return dimension_array_names


def create_dimension_arrays_from_coordinates(
    prefetch_dataset: Dataset,
    latitude_coordinate: VariableFromDmr,
    longitude_coordinate: VariableFromDmr,
    crs: CRS,
    projected_dimension_names: list[str],
) -> dict[str, np.ndarray]:
    """Generate artificial 1D dimensions scales for each
    2D dimension or coordinate variable.
    1) Get 2 valid geo grid points
    2) convert them to a projected x-y extent
    3) Generate the x-y dimscale array and return to the calling method

    """
    if len(projected_dimension_names) < 2:
        raise InvalidDimensionNames(projected_dimension_names)

    lat_arr = get_2d_coordinate_array(
        prefetch_dataset,
        latitude_coordinate.full_name_path,
    )
    lon_arr = get_2d_coordinate_array(
        prefetch_dataset,
        longitude_coordinate.full_name_path,
    )

    row_indices, col_indices = get_valid_sample_pts(
        lat_arr, lon_arr, latitude_coordinate, longitude_coordinate
    )

    dim_order_is_y_x, row_dim_values = get_dimension_order_and_dim_values(
        lat_arr, lon_arr, row_indices, crs, is_row=True
    )
    dim_order, col_dim_values = get_dimension_order_and_dim_values(
        lat_arr, lon_arr, col_indices, crs, is_row=False
    )
    if dim_order_is_y_x != dim_order:
        raise InvalidCoordinateData("the order of dimensions do not match")

    row_size, col_size = get_row_col_sizes_from_coordinates(
        lat_arr, lon_arr, dim_order_is_y_x
    )

    y_dim = interpolate_dim_values_from_sample_pts(
        row_dim_values, np.transpose(row_indices)[0], row_size
    )

    x_dim = interpolate_dim_values_from_sample_pts(
        col_dim_values, np.transpose(col_indices)[1], col_size
    )

    projected_y, projected_x = (
        projected_dimension_names[-2],
        projected_dimension_names[-1],
    )

    if dim_order_is_y_x:
        return {projected_y: y_dim, projected_x: x_dim}
    raise UnsupportedDimensionOrder('x,y')
    # this is not currently supported in the calling function in spatial.py
    # return {projected_x: x_dim, projected_y: y_dim}


def get_2d_coordinate_array(
    prefetch_dataset: Dataset,
    coordinate_name: str,
) -> np.ndarray:
    """This function returns the `numpy` array from a
    coordinate dataset.

    """
    try:
        coordinate_array = prefetch_dataset[coordinate_name][:]
    except IndexError as exception:
        raise MissingCoordinateVariable(coordinate_name) from exception

    return coordinate_array


def get_valid_sample_pts(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    lat_coordinate: VariableFromDmr,
    lon_coordinate: VariableFromDmr,
) -> tuple[list, list]:
    """
    This function finds a set of indices maximally spread across
    a row, and the set maximally spread across a column, with the
    indices being valid in both the latitude and longitude datasets.
    When interpolating between these points, the maximal spread
    ensures the greatest interpolation accuracy.
    """
    valid_lat_lon_mask = np.logical_and(
        get_valid_indices(lat_arr, lat_coordinate),
        get_valid_indices(lon_arr, lon_coordinate),
    )

    # get maximally spread points within rows
    max_x_spread_pts = get_max_spread_pts(~valid_lat_lon_mask)

    # Doing the same for the columns is done by transposing the valid_mask
    # and then fixing the results from [x, y] to [y, x].
    max_y_spread_trsp = get_max_spread_pts(np.transpose(~valid_lat_lon_mask))
    max_y_spread_pts = [
        list(np.flip(max_y_spread_trsp[0])),
        list(np.flip(max_y_spread_trsp[1])),
    ]

    return max_y_spread_pts, max_x_spread_pts


def get_valid_indices(
    lat_lon_array: np.ndarray, coordinate: VariableFromDmr
) -> np.ndarray:
    """
    Returns an array of boolean values indicating valid values - non-fill,
    within range - for a given coordinate variable. Returns an empty
    ndarray of size (0,0) for any other variable.
    """

    # get_attribute_value returns a value of type `str`
    coordinate_fill = coordinate.get_attribute_value('_FillValue')
    if coordinate_fill is not None:
        is_not_fill = ~np.isclose(lat_lon_array, float(coordinate_fill))
    else:
        # Creates an entire array of `True` values.
        is_not_fill = np.ones_like(lat_lon_array, dtype=bool)

    if coordinate.is_longitude():
        valid_indices = np.logical_and(
            is_not_fill,
            np.logical_and(lat_lon_array >= -180.0, lat_lon_array <= 360.0),
        )
    elif coordinate.is_latitude():
        valid_indices = np.logical_and(
            is_not_fill,
            np.logical_and(lat_lon_array >= -90.0, lat_lon_array <= 90.0),
        )
    else:
        raise InvalidCoordinateDataset(coordinate.full_name_path)

    return valid_indices


def get_max_spread_pts(
    valid_geospatial_mask: np.ndarray,
) -> list[list]:
    """
    This function returns two data points by x, y indices that are spread farthest
    from each other in the same row, i.e., have the greatest delta-x value - and
    are valid data points from the valid_geospatial_mask array passed in. The input array
    must be a 2D Numpy mask array providing the valid data points, e.g., filtering
    out fill values and out-of-range values.
    - input is Numpy Mask Array, e.g., invalid latitudes & longitudes
    - returns 2 points by indices, [[y_ind, x_ind], [y_ind, x_ind]
    """
    # fill a sample array with index values, arr_ind[i, j] = j
    arr_indices = np.indices(
        (valid_geospatial_mask.shape[-2], valid_geospatial_mask.shape[-1])
    )[1]
    if valid_geospatial_mask.ndim == 2:
        # mask arr_ind to hide the invalid data points
        valid_indices = np.ma.array(arr_indices, mask=valid_geospatial_mask)
    elif valid_geospatial_mask.ndim == 3:
        # use just 2 of the dimensions
        # mask arr_ind to hide the invalid data points
        valid_indices = np.ma.array(arr_indices, mask=valid_geospatial_mask[0, :, :])
    else:
        raise NotImplementedError

    if valid_indices.count() == 0:
        raise InvalidCoordinateData("No valid coordinate data")

    # ptp (peak-to-peak) finds the greatest delta-x value amongst valid points
    # for each row. Result is 1D
    index_spread = valid_indices.ptp(axis=1)

    # This finds which row has the greatest spread (delta-x)
    max_spread = np.argmax(index_spread)

    # Using the row reference, find the min and max
    min_index = np.min(valid_indices[max_spread])
    max_index = np.max(valid_indices[max_spread])

    # There is just one valid point
    if min_index == max_index:
        raise InvalidCoordinateData("Only one valid point in coordinate data")

    return [[max_spread, min_index], [max_spread, max_index]]


def get_dimension_order_and_dim_values(
    lat_array_points: np.ndarray,
    lon_array_points: np.ndarray,
    grid_dimension_indices: list[tuple[int, int]],
    crs: CRS,
    is_row: bool,
) -> tuple[bool, np.ndarray]:
    """Determines the order of dimensions based on whether the
    projected y or projected_x values are varying across row or column.
    Also returns a 1-D array of dimension values for the requested
    projected spatial dimension. The input lat lon arrays and dimension
    indices are assumed to be 2D in this implementation of the function.
    """
    if lat_array_points.ndim == 1 and lon_array_points.ndim == 1:
        lat_arr_values = lat_array_points
        lon_arr_values = lon_array_points
    elif lat_array_points.ndim == 2 and lon_array_points.ndim == 2:
        lat_arr_values = [lat_array_points[i][j] for i, j in grid_dimension_indices]
        lon_arr_values = [lon_array_points[i][j] for i, j in grid_dimension_indices]
    else:
        # assuming a nominal z,y,x order
        lat_arr_values = [lat_array_points[0][i][j] for i, j in grid_dimension_indices]
        lon_arr_values = [lon_array_points[0][i][j] for i, j in grid_dimension_indices]

    from_geo_transformer = Transformer.from_crs(4326, crs)
    x_values, y_values = (  # pylint: disable=unpacking-non-sequence
        from_geo_transformer.transform(lat_arr_values, lon_arr_values)
    )
    y_variance = np.abs(np.diff(y_values))
    x_variance = np.abs(np.diff(x_values))

    # If input lat/lon array is row and projected y_values is varying more
    # than projected x_values, then the dimensions are ordered (y,x)
    # If input lat/lon array is col and projected y_values is changing more
    # than projected x_values, then the dimensions are ordered (x,y)

    if y_variance > x_variance:
        is_y_x_order = is_row
        dimension_values = y_values
    elif x_variance > y_variance:
        is_y_x_order = not is_row
        dimension_values = x_values
    else:
        raise InvalidCoordinateData("x/y values are varying by the same amount")

    return is_y_x_order, dimension_values


def get_row_col_sizes_from_coordinates(
    lat_arr: np.ndarray, lon_arr: np.ndarray, dim_order_is_y_x: bool
) -> tuple[int, int]:
    """
    This function returns the row and column sizes of the coordinate datasets
    The last two dimensions of the array correspond to the spatial dimensions.
    which is the recommendation from CF-Conventions.
    """

    if lat_arr.ndim >= 2 and lon_arr.shape == lat_arr.shape:
        col_size = lat_arr.shape[-1]
        row_size = lat_arr.shape[-2]
    elif (
        lat_arr.ndim == 1
        and lon_arr.ndim == 1
        and lat_arr.size > 0
        and lon_arr.size > 0
    ):
        if dim_order_is_y_x:
            col_size = lon_arr.size
            row_size = lat_arr.size
        else:
            col_size = lat_arr.size
            row_size = lon_arr.size
    else:
        raise IncompatibleCoordinateVariables(lon_arr.shape, lat_arr.shape)
    return row_size, col_size


def interpolate_dim_values_from_sample_pts(
    dim_values: np.ndarray,
    dim_indices: list[int],
    dim_size: int,
) -> np.ndarray:
    """
    Return a full dimension data array based upon 2 valid projected values
    given in dim_values and located by dim_indices. The dim_indices need
    to be between 0 and dim_size. Returns a 1D array of size = dim_size
    with proper dimension array values, with linear interpolation between
    the given dim_values.
    """

    if (dim_indices[1] != dim_indices[0]) and (dim_values[1] != dim_values[0]):
        dim_resolution = (dim_values[1] - dim_values[0]) / (
            dim_indices[1] - dim_indices[0]
        )
    else:
        raise InvalidCoordinateData(
            'No distinct valid coordinate points - '
            f'dim_index={dim_indices[0]}, dim_value={dim_values[0]}'
        )

    dim_min = dim_values[0] - (dim_resolution * dim_indices[0])
    dim_max = dim_values[1] + (dim_resolution * (dim_size - 1 - dim_indices[1]))

    return np.linspace(dim_min, dim_max, dim_size)
