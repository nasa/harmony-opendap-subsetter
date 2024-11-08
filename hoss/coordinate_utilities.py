""" This module contains utility functions used for
    coordinate variables and functions to convert the
    coordinate variable data to projected x/y dimension values
"""

from typing import Dict

import numpy as np
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from varinfo import VariableFromDmr, VarInfoFromDmr

from hoss.exceptions import (
    IncompatibleCoordinateVariables,
    InvalidCoordinateDataset,
    InvalidCoordinateVariable,
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
    """This function returns latitude and longitude variables listed in the
    CF-Convention coordinates metadata attribute. It returns them in a specific
    order [latitude, longitude]"
    """
    # varinfo returns a set and not a list
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
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
) -> tuple[int, int]:
    """
    This function returns the row and column sizes of the coordinate datasets

    """
    # ToDo - if the coordinates are 3D
    if lat_arr.ndim > 1 and lon_arr.shape == lat_arr.shape:
        col_size = lat_arr.shape[1]
        row_size = lat_arr.shape[0]
    elif (
        lat_arr.ndim == 1
        and lon_arr.ndim == 1
        and lat_arr.size > 0
        and lon_arr.size > 0
    ):
        # Todo: The ordering needs to be checked
        col_size = lon_arr.size
        row_size = lat_arr.size
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


def get_1D_dim_array_data_from_dimvalues(
    dim_values: np.ndarray, dim_indices: np.ndarray, dim_size: int
) -> np.ndarray:
    """
    return a full dimension data array based on the 2 projected points and
    grid size
    """

    if (dim_indices[1] != dim_indices[0]) and (dim_values[1] != dim_values[0]):
        dim_resolution = (dim_values[1] - dim_values[0]) / (
            dim_indices[1] - dim_indices[0]
        )
    else:
        raise InvalidCoordinateDataset(dim_values[0], dim_indices[0])

    dim_min = dim_values[0] - (dim_resolution * dim_indices[0])
    dim_max = dim_values[1] + (dim_resolution * (dim_size - 1 - dim_indices[1]))

    return np.linspace(dim_min, dim_max, dim_size)


def get_valid_indices(
    coordinate_row_col: np.ndarray, coordinate: VariableFromDmr
) -> np.ndarray:
    """
    Returns indices of a valid array without fill values if the fill
    value is provided. If it is not provided, we check for valid values
    for latitude and longitude
    """

    # get_attribute_value returns a value of type `str`
    coordinate_fill = coordinate.get_attribute_value('_FillValue')
    if coordinate_fill is not None:
        is_not_fill = ~np.isclose(coordinate_row_col, float(coordinate_fill))
    else:
        # Creates an entire array of `True` values.
        is_not_fill = np.ones_like(coordinate_row_col, dtype=bool)

    if coordinate.is_longitude():
        valid_indices = np.where(
            np.logical_and(
                is_not_fill,
                np.logical_and(
                    coordinate_row_col >= -180.0, coordinate_row_col <= 360.0
                ),
            )
        )[0]
    elif coordinate.is_latitude():
        valid_indices = np.where(
            np.logical_and(
                is_not_fill,
                np.logical_and(coordinate_row_col >= -90.0, coordinate_row_col <= 90.0),
            )
        )[0]
    else:
        valid_indices = np.empty((0, 0))

    return valid_indices


def get_row_col_geo_grid_points(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    latitude_coordinate: VariableFromDmr,
    longitude_coordinate: VariableFromDmr,
    row_size: float,
    col_size: float,
) -> tuple[dict, dict]:
    """
    This method is used to return two valid lat lon points from a 2D
    coordinate dataset. It gets the row and column of the latitude and longitude
    arrays to get two valid points. This does a check for fill values and
    This method does not go down to the next row and col. if the selected row and
    column all have fills, it will raise an exception in those cases.
    """

    geo_grid_indexes_row, geo_grid_indexes_col = get_row_col_valid_indices_in_dataset(
        lat_arr, lon_arr, row_size, col_size, latitude_coordinate, longitude_coordinate
    )
    # 2 points in same row - column indices are changing
    # point1_row_index, point1_col_index = geo_grid_indexes_row[0]
    # point2_row_index, point2_col_index = geo_grid_indexes_row[1]

    # 2 points in same column - row indices are changing
    # point3_row_index, point3_col_index = geo_grid_indexes_col[0]
    # point4_row_index, point4_col_index = geo_grid_indexes_col[1]

    geo_grid_col_points = [
        (
            lon_arr[geo_grid_indexes_row[0][0]][geo_grid_indexes_row[0][1]],
            lat_arr[geo_grid_indexes_row[0][0]][geo_grid_indexes_row[0][1]],
        ),
        (
            lon_arr[geo_grid_indexes_row[1][0]][geo_grid_indexes_row[1][1]],
            lat_arr[geo_grid_indexes_row[1][0]][geo_grid_indexes_row[1][1]],
        ),
    ]

    geo_grid_row_points = [
        (
            # lon_arr[point3_row_index][point3_col_index],
            # lat_arr[point3_row_index][point3_col_index],
            lon_arr[geo_grid_indexes_col[0][0]][geo_grid_indexes_col[0][1]],
            lat_arr[geo_grid_indexes_col[0][0]][geo_grid_indexes_col[0][1]],
        ),
        (
            # lon_arr[point4_row_index][point4_col_index],
            # lat_arr[point4_row_index][point4_col_index],
            lon_arr[geo_grid_indexes_col[1][0]][geo_grid_indexes_col[1][1]],
            lat_arr[geo_grid_indexes_col[1][0]][geo_grid_indexes_col[1][1]],
        ),
    ]
    return (
        {
            geo_grid_indexes_col[0]: geo_grid_row_points[0],
            geo_grid_indexes_col[1]: geo_grid_row_points[1],
        },
        {
            geo_grid_indexes_row[0]: geo_grid_col_points[0],
            geo_grid_indexes_row[1]: geo_grid_col_points[1],
        },
    )


def get_x_y_values_from_geographic_points(points: Dict, crs: CRS) -> Dict[tuple, tuple]:
    """Take an input list of (longitude, latitude) coordinates and project
    those points to the target grid. Then return the x-y dimscales

    """
    point_longitudes, point_latitudes = zip(*list(points.values()))

    from_geo_transformer = Transformer.from_crs(4326, crs)
    points_x, points_y = (  # pylint: disable=unpacking-non-sequence
        from_geo_transformer.transform(point_latitudes, point_longitudes)
    )

    x_y_points = {}
    for index, point_x, point_y in zip(list(points.keys()), points_x, points_y):
        x_y_points.update({index: (point_x, point_y)})

    return x_y_points


def get_row_col_valid_indices_in_dataset(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    row_size: int,
    col_size: int,
    lat_coordinate: VariableFromDmr,
    lon_coordinate: VariableFromDmr,
) -> tuple[list, list]:
    """
    This method gets valid indices in a row or column of a
    coordinate dataset
    """
    # coordinate_index = start_index
    valid_row_indices = valid_col_indices = np.empty((0, 0))
    # if span_type == 'row':
    # get 2 points in the row
    # while valid_col_indices.size < 2:
    # if span_type == 'row':
    max_col_distance = 0
    valid_row_index = row_coordinate_index = -1
    max_range_valid_col_indices = np.empty(0)
    while row_coordinate_index < row_size - 1:
        row_coordinate_index = row_coordinate_index + 1
        valid_col_indices = np.intersect1d(
            get_valid_indices(lat_arr[row_coordinate_index, :], lat_coordinate),
            get_valid_indices(lon_arr[row_coordinate_index, :], lon_coordinate),
        )

        if valid_col_indices.size >= 2 and (
            max_col_distance < (valid_col_indices[-1] - valid_col_indices[0])
        ):
            max_col_distance = valid_col_indices[-1] - valid_col_indices[0]
            max_range_valid_col_indices = valid_col_indices
            valid_row_index = row_coordinate_index

    if valid_row_index < 0 or max_range_valid_col_indices.size < 2:
        raise InvalidCoordinateVariable(lat_coordinate.full_name_path)

    # span type is col
    # get 2 points in a column
    # while valid_row_indices.size < 2:
    max_row_distance = 0
    valid_col_index = col_coordinate_index = col_size
    max_range_valid_row_indices = np.empty(0)
    while col_coordinate_index > 0:
        col_coordinate_index = col_coordinate_index - 1
        valid_row_indices = np.intersect1d(
            get_valid_indices(lat_arr[:, col_coordinate_index], lat_coordinate),
            get_valid_indices(lon_arr[:, col_coordinate_index], lon_coordinate),
        )
        if valid_row_indices.size >= 2 and (
            max_row_distance < (valid_row_indices[-1] - valid_row_indices[0])
        ):
            max_row_distance = valid_row_indices[-1] - valid_row_indices[0]
            max_range_valid_row_indices = valid_row_indices
            valid_col_index = col_coordinate_index

    if col_coordinate_index >= col_size or max_range_valid_row_indices.size < 2:
        raise InvalidCoordinateVariable(lon_coordinate.full_name_path)

    # same row . 2 column points
    geo_grid_indexes_row = [
        (valid_row_index, max_range_valid_col_indices[0]),
        (valid_row_index, max_range_valid_col_indices[-1]),
    ]

    # same column, 2 row points
    geo_grid_indexes_col = [
        (max_range_valid_row_indices[0], valid_col_index),
        (max_range_valid_row_indices[-1], valid_col_index),
    ]

    return geo_grid_indexes_row, geo_grid_indexes_col


def create_dimension_array_from_coordinates(
    prefetch_dataset: Dataset,
    latitude_coordinate: VariableFromDmr,
    longitude_coordinate: VariableFromDmr,
    crs: CRS,
    projected_dimension_names: list,
) -> Dict[str, np.ndarray]:
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

    row_size, col_size = get_row_col_sizes_from_coordinate_datasets(
        lat_arr,
        lon_arr,
    )

    geo_grid_row_points, geo_grid_col_points = get_row_col_geo_grid_points(
        lat_arr, lon_arr, latitude_coordinate, longitude_coordinate, row_size, col_size
    )

    x_y_values1 = get_x_y_values_from_geographic_points(geo_grid_row_points, crs)
    # y value changes across the row indices
    row_indices = [list(x_y_values1.keys())[0][0], list(x_y_values1.keys())[1][0]]
    y_values = [list(x_y_values1.values())[0][1], list(x_y_values1.values())[1][1]]

    x_y_values2 = get_x_y_values_from_geographic_points(geo_grid_col_points, crs)
    # x value changes across the col indices
    col_indices = [list(x_y_values2.keys())[0][1], list(x_y_values2.keys())[1][1]]
    x_values = [list(x_y_values2.values())[0][0], list(x_y_values2.values())[1][0]]

    y_dim = get_1D_dim_array_data_from_dimvalues(y_values, row_indices, row_size)
    x_dim = get_1D_dim_array_data_from_dimvalues(x_values, col_indices, col_size)

    projected_y, projected_x = tuple(projected_dimension_names)
    return {projected_y: y_dim, projected_x: x_dim}
