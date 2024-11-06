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
    # print(f'coordinate_row_col={coordinate_row_col}')
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
    print(f'valid_indices={valid_indices}')
    return valid_indices


def get_two_valid_geo_grid_points(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    latitude_coordinate: VariableFromDmr,
    longitude_coordinate: VariableFromDmr,
    row_size: float,
    col_size: float,
) -> dict[int, tuple]:
    """
    This method is used to return two valid lat lon points from a 2D
    coordinate dataset. It gets the row and column of the latitude and longitude
    arrays to get two valid points. This does a check for fill values and
    This method does not go down to the next row and col. if the selected row and
    column all have fills, it will raise an exception in those cases.
    """
    first_row_col_index = -1
    first_row_row_index = 0
    last_col_row_index = -1
    last_col_col_index = col_size - 1
    lat_row_valid_indices = lon_row_valid_indices = np.empty((0, 0))

    # get the first row with points that are valid in the lat and lon rows
    first_row_row_index, lat_row_valid_indices = get_valid_indices_in_dataset(
        lat_arr, row_size, latitude_coordinate, 'row', first_row_row_index
    )
    first_row_row_index1, lon_row_valid_indices = get_valid_indices_in_dataset(
        lon_arr, row_size, longitude_coordinate, 'row', first_row_row_index
    )
    # get a point that is common on both row datasets
    if (
        (first_row_row_index == first_row_row_index1)
        and (lat_row_valid_indices.size > 0)
        and (lon_row_valid_indices.size > 0)
    ):
        first_row_col_index = np.intersect1d(
            lat_row_valid_indices, lon_row_valid_indices
        )[0]

        print(f'first_row_row_index={first_row_row_index}')
        print(f'first_row_col_index={first_row_col_index}')

    # get a valid column from the latitude and longitude datasets
    last_col_col_index, lon_col_valid_indices = get_valid_indices_in_dataset(
        lon_arr, col_size, longitude_coordinate, 'col', last_col_col_index
    )
    last_col_col_index1, lat_col_valid_indices = get_valid_indices_in_dataset(
        lat_arr, col_size, latitude_coordinate, 'col', last_col_col_index
    )

    # get a point that is common to both column datasets
    if (
        (last_col_col_index == last_col_col_index1)
        and (lat_col_valid_indices.size > 0)
        and (lon_col_valid_indices.size > 0)
    ):
        last_col_row_index = np.intersect1d(
            lat_col_valid_indices, lon_col_valid_indices
        )[-1]

        print(f'last_col_col_index={last_col_col_index}')
        print(f'last_col_row_index={last_col_row_index}')

    # if the whole row and whole column has no valid indices
    # we throw an exception now. This can be extended to move
    # to the next row/col
    if first_row_col_index == -1:
        raise InvalidCoordinateVariable('latitude/longitude')
    if last_col_row_index == -1:
        raise InvalidCoordinateVariable('latitude/longitude')

    geo_grid_indexes = [
        (first_row_row_index, first_row_col_index),
        (last_col_row_index, last_col_col_index),
    ]

    geo_grid_points = [
        (
            lon_arr[first_row_row_index][first_row_col_index],
            lat_arr[first_row_row_index][first_row_col_index],
        ),
        (
            lon_arr[last_col_row_index][last_col_col_index],
            lat_arr[last_col_row_index][last_col_col_index],
        ),
    ]

    return {
        geo_grid_indexes[0]: geo_grid_points[0],
        geo_grid_indexes[1]: geo_grid_points[1],
    }


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


def get_valid_indices_in_dataset(
    coordinate_arr: np.ndarray,
    dim_size: int,
    coordinate: VariableFromDmr,
    span_type: str,
    start_index: int,
) -> tuple[int, np.ndarray]:
    """
    This method gets valid indices in a row or column of a
    coordinate dataset
    """
    coordinate_index = start_index
    valid_indices = []
    if span_type == 'row':
        valid_indices = get_valid_indices(
            coordinate_arr[coordinate_index, :], coordinate
        )
    else:
        valid_indices = get_valid_indices(
            coordinate_arr[:, coordinate_index], coordinate
        )

    while valid_indices.size == 0:
        if span_type == 'row':
            if coordinate_index < dim_size:
                coordinate_index = coordinate_index + 1
                valid_indices = get_valid_indices(
                    coordinate_arr[coordinate_index, :],
                    coordinate,
                )
            else:
                raise InvalidCoordinateVariable(coordinate.full_name_path)
        else:
            if coordinate_index > 0:
                coordinate_index = coordinate_index - 1
                valid_indices = get_valid_indices(
                    coordinate_arr[:, coordinate_index], coordinate
                )
            else:
                raise InvalidCoordinateVariable(coordinate.full_name_path)
    return coordinate_index, valid_indices


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
    geo_grid_points = get_two_valid_geo_grid_points(
        lat_arr, lon_arr, latitude_coordinate, longitude_coordinate, row_size, col_size
    )

    x_y_values = get_x_y_values_from_geographic_points(geo_grid_points, crs)

    row_indices, col_indices = zip(*list(x_y_values.keys()))

    x_values, y_values = zip(*list(x_y_values.values()))

    y_dim = get_1D_dim_array_data_from_dimvalues(y_values, row_indices, row_size)

    x_dim = get_1D_dim_array_data_from_dimvalues(x_values, col_indices, col_size)

    projected_y, projected_x = tuple(projected_dimension_names)
    return {projected_y: y_dim, projected_x: x_dim}
