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
    MissingCoordinateVariable,
    MissingVariable,
)


def get_projected_dimension_names(varinfo: VarInfoFromDmr, variable_name: str) -> str:
    """returns the x-y projection variable names that would
    match the group of the input variable. The 'projected_y' dimension
    and 'projected_x' names are returned with the group pathname

    """
    variable = varinfo.get_variable(variable_name)

    if variable is not None:
        projected_dimension_names = [
            f'{variable.group_path}/projected_y',
            f'{variable.group_path}/projected_x',
        ]
    else:
        raise MissingVariable(variable_name)

    return projected_dimension_names


def get_projected_dimension_names_from_coordinate_variables(
    varinfo: VarInfoFromDmr,
    variable_name: str,
) -> list[str]:
    """
    Returns the projected dimensions names from coordinate variables
    """
    latitude_coordinates, longitude_coordinates = get_coordinate_variables(
        varinfo, [variable_name]
    )

    if len(latitude_coordinates) == 1 and len(longitude_coordinates) == 1:
        projected_dimension_names = get_projected_dimension_names(
            varinfo, latitude_coordinates[0]
        )

    # if the override is the variable
    elif (
        varinfo.get_variable(variable_name).is_latitude()
        or varinfo.get_variable(variable_name).is_longitude()
    ):
        projected_dimension_names = get_projected_dimension_names(
            varinfo, variable_name
        )
    else:
        projected_dimension_names = []
    return projected_dimension_names


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
        if (len(varinfo.get_variable(variable).dimensions) == 0)
        or (any_absent_dimension_variables(varinfo, variable))
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


def get_coordinate_variables(
    varinfo: VarInfoFromDmr,
    requested_variables: list,
) -> tuple[list, list]:
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


def get_row_col_sizes_from_coordinate_datasets(
    lat_arr: np.ndarray, lon_arr: np.ndarray, dim_order_is_y_x: bool
) -> tuple[int, int]:
    """
    This function returns the row and column sizes of the coordinate datasets

    """
    # ToDo - if the coordinates are 3D
    if lat_arr.ndim == 2 and lon_arr.shape == lat_arr.shape:
        col_size = lat_arr.shape[1]
        row_size = lat_arr.shape[0]
    elif lat_arr.ndim > 2 and lon_arr.shape == lat_arr.shape:
        raise InvalidCoordinateData("Greater than 2D is not supported yet")
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


def get_coordinate_array(
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


def get_1d_dim_array_data_from_dimvalues(
    dim_values: np.ndarray,
    dim_indices: np.ndarray,
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


def get_x_y_values_from_geographic_points(
    points: list[tuple],
    crs: CRS,
) -> list[tuple]:
    """Takes an input list of (longitude, latitude) coordinates and CRS object
    from PyProj and projects those points to the target grid. Then returns
    the list of x-y points as tuple in (x,y) order

    """
    point_longitudes, point_latitudes = zip(*list(points))

    from_geo_transformer = Transformer.from_crs(4326, crs)
    points_x, points_y = (  # pylint: disable=unpacking-non-sequence
        from_geo_transformer.transform(point_latitudes, point_longitudes)
    )

    return list(zip(points_x, points_y))


def get_valid_row_col_pairs(
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
        (valid_geospatial_mask.shape[0], valid_geospatial_mask.shape[1])
    )[1]

    # mask arr_ind to hide the invalid data points
    valid_indices = np.ma.array(arr_indices, mask=valid_geospatial_mask)

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


def create_dimension_arrays_from_coordinates(
    prefetch_dataset: Dataset,
    latitude_coordinate: VariableFromDmr,
    longitude_coordinate: VariableFromDmr,
    crs: CRS,
    projected_dimension_names: list,
) -> dict[str, np.ndarray]:
    """Generate artificial 1D dimensions scales for each
    2D dimension or coordinate variable.
    1) Get 2 valid geo grid points
    2) convert them to a projected x-y extent
    3) Generate the x-y dimscale array and return to the calling method

    """
    lat_arr = get_coordinate_array(
        prefetch_dataset,
        latitude_coordinate.full_name_path,
    )
    lon_arr = get_coordinate_array(
        prefetch_dataset,
        longitude_coordinate.full_name_path,
    )


    row_indices, col_indices = get_valid_row_col_pairs(
        lat_arr, lon_arr, latitude_coordinate, longitude_coordinate
    )


    dim_order_is_y_x = get_dimension_order(lat_arr, lon_arr, row_indices, is_row=True)
    dim_order = get_dimension_order(lat_arr, lon_arr, col_indices, is_row=False)
    if dim_order_is_y_x != dim_order:
        raise InvalidCoordinateData("the order of dimensions do not match")

    row_size, col_size = get_row_col_sizes_from_coordinate_datasets(
        lat_arr, lon_arr, dim_order_is_y_x
    )

    y_dim = get_dimension_array_from_geo_points(
        lat_arr,
        lon_arr,
        crs,
        row_indices,
        row_size,
        dim_order_is_y_x,
        use_row_not_col=True,
    )

    x_dim = get_dimension_array_from_geo_points(
        lat_arr,
        lon_arr,
        crs,
        col_indices,
        col_size,
        dim_order_is_y_x,
        use_row_not_col=False,
    )
    projected_y, projected_x = tuple(projected_dimension_names)

    if dim_order_is_y_x:
        return {projected_y: y_dim, projected_x: x_dim}
    return {projected_x: x_dim, projected_y: y_dim}



def get_dimension_array_from_geo_points(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    crs: CRS,
    dimension_indices: list,
    dimension_size: int,
    dim_order_is_y_x: bool,
    use_row_not_col: bool,

) -> np.ndarray:
    """This function uses the list of lat/lon points corresponding
    to a list of array indices and reprojects it with the CRS
    provided and scales the x/y values to a dimension array with the dimension
    size provided. index_for_dimension is 0 for row and
    1 for column. x_or_y_index is 1 if latitude is used for row
    and is 0 if longitude is used for row. (it determines the dimension order)

    """
    if use_row_not_col:
        index_for_dimension = 0
        if dim_order_is_y_x:
            x_or_y_index = 1
        else:
            x_or_y_index = 0
    else:
        index_for_dimension = 1
        if dim_order_is_y_x:
            x_or_y_index = 0
        else:
            x_or_y_index = 1


    geo_grid_points = [
        (lon_arr[row, col], lat_arr[row, col]) for row, col in dimension_indices
    ]


    x_y_values = get_x_y_values_from_geographic_points(geo_grid_points, crs)

    indices_for_dimension = np.transpose(dimension_indices)[index_for_dimension]

    dimension_values = np.transpose(x_y_values)[x_or_y_index]

    return get_1d_dim_array_data_from_dimvalues(
        dimension_values, indices_for_dimension, dimension_size
    )


def get_dimension_order(
    lat_array_points: np.ndarray,
    lon_array_points: np.ndarray,
    dimension_indices: list,
    is_row: bool,
) -> bool:
    """Determines the order of dimensions based on whether the
    latitude and longitude are varying across row or column
    """

    # if lat/lon array is 2D and variables are also 2D
    lat_arr_values = [lat_array_points[i][j] for i, j in dimension_indices]
    lon_arr_values = [lon_array_points[i][j] for i, j in dimension_indices]

    # if it is row and lat is changing y_x order is true
    if (is_row is True) and (np.allclose(lat_arr_values, lat_arr_values[0]) is False):
        return True

    # if it is row and lon is changing return y_x order is false
    if (is_row is True) and (np.allclose(lon_arr_values, lon_arr_values[0]) is False):
        return False

    # if it is col and lat is changing y_x order is false
    if (is_row is False) and (np.allclose(lat_arr_values, lat_arr_values[0]) is False):
        return False

    # if it is col and lon is changing y_x order is true
    if (is_row is False) and (np.allclose(lon_arr_values, lon_arr_values[0]) is False):
        return True

    raise InvalidCoordinateData("lat/lon values are constant")

