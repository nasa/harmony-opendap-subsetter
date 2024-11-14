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


def get_dim_array_data_from_dimvalues(
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
    lat_lon_array: np.ndarray, coordinate: VariableFromDmr
) -> np.ndarray:
    """
    Returns indices of a valid array without fill values if the fill
    value is provided. If it is not provided, we check for valid values
    for latitude and longitude
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
        valid_indices = np.empty((0, 0))

    return valid_indices  # throw an exception


def get_row_col_geo_grid_points(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    latitude_coordinate: VariableFromDmr,
    longitude_coordinate: VariableFromDmr,
) -> tuple[dict, dict]:
    """
    This method is used to return two valid lat lon points from a 2D
    coordinate dataset. It gets the row and column of the latitude and longitude
    arrays to get two valid points. This does a check for fill values and
    This method does not go down to the next row and col. if the selected row and
    column all have fills, it will raise an exception in those cases.
    """

    geo_grid_indices_row, geo_grid_indices_col = get_row_col_valid_indices_in_dataset(
        lat_arr, lon_arr, latitude_coordinate, longitude_coordinate
    )

    geo_grid_col_points = [
        (
            lon_arr[geo_grid_indices_row[0][0]][geo_grid_indices_row[0][1]],
            lat_arr[geo_grid_indices_row[0][0]][geo_grid_indices_row[0][1]],
        ),
        (
            lon_arr[geo_grid_indices_row[1][0]][geo_grid_indices_row[1][1]],
            lat_arr[geo_grid_indices_row[1][0]][geo_grid_indices_row[1][1]],
        ),
    ]

    geo_grid_row_points = [
        (
            lon_arr[geo_grid_indices_col[0][0]][geo_grid_indices_col[0][1]],
            lat_arr[geo_grid_indices_col[0][0]][geo_grid_indices_col[0][1]],
        ),
        (
            lon_arr[geo_grid_indices_col[1][0]][geo_grid_indices_col[1][1]],
            lat_arr[geo_grid_indices_col[1][0]][geo_grid_indices_col[1][1]],
        ),
    ]

    return (
        {
            (
                geo_grid_indices_col[0][0],
                geo_grid_indices_col[0][1],
            ): geo_grid_row_points[0],
            (
                geo_grid_indices_col[1][0],
                geo_grid_indices_col[1][1],
            ): geo_grid_row_points[1],
        },
        {
            (
                geo_grid_indices_row[0][0],
                geo_grid_indices_row[0][1],
            ): geo_grid_col_points[0],
            (
                geo_grid_indices_row[1][0],
                geo_grid_indices_row[1][1],
            ): geo_grid_col_points[1],
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
    lat_coordinate: VariableFromDmr,
    lon_coordinate: VariableFromDmr,
) -> tuple[list, list]:
    """
    This function gets valid indices across rows and columns of
    both the latitude and longitude datasets
    """
    valid_lat_lon_mask = np.logical_and(
        get_valid_indices(lat_arr, lat_coordinate),
        get_valid_indices(lon_arr, lon_coordinate),
    )

    # get maximally spread points within rows
    max_x_spread_pts = get_max_x_spread_pts(~valid_lat_lon_mask)

    # Doing the same for the columns is done by transposing the valid_mask
    # and then fixing the results from [x, y] to [y, x].
    max_y_spread_trsp = get_max_x_spread_pts(np.transpose(~valid_lat_lon_mask))
    max_y_spread_pts = [
        np.flip(max_y_spread_trsp[0]),
        np.flip(max_y_spread_trsp[1]),
    ]

    return max_y_spread_pts, max_x_spread_pts


def get_max_x_spread_pts(
    valid_mask: np.ndarray,  # Numpy Mask Array, e.g., invalid latitudes & longitudes
) -> list[list]:  # 2 points by indices, [[y_ind, x_ind], [y_ind, x_ind]
    """
    # This function returns two data points by x, y indices that are spread farthest
    # from each other in the same row, i.e., have the greatest delta-x value - and
    # are valid data points from the valid_mask array passed in. The input array
    # must be a 2D Numpy mask array providing the valid data points, e.g., filtering
    # out fill values and out-of-range values.
    """
    # fill a sample array with x-index values, x_ind[i, j] = j
    x_ind = np.array(
        [
            # [j for j in range(valid_mask.shape[1])]
            list(range(valid_mask.shape[1]))
            for i in range(valid_mask.shape[0])
        ]
    )
    # mask x_ind to hide the invalid data points
    valid_x_ind = np.ma.array(x_ind, mask=valid_mask)

    # ptp (peak-to-peak) finds the greatest delta-x value amongst valid points
    # for each row. Result is 1D
    x_ind_spread = valid_x_ind.ptp(axis=1)

    # This finds which row has the greatest spread (delta-x)
    max_x_spread_row = np.argmax(x_ind_spread)

    # Using the row reference, find the min-x and max-x
    min_x_ind = np.min(valid_x_ind[max_x_spread_row])
    max_x_ind = np.max(valid_x_ind[max_x_spread_row])

    return [[max_x_spread_row, min_x_ind], [max_x_spread_row, max_x_ind]]


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

    geo_grid_col_points, geo_grid_row_points = get_row_col_geo_grid_points(
        lat_arr, lon_arr, latitude_coordinate, longitude_coordinate
    )
    x_y_values1 = get_x_y_values_from_geographic_points(geo_grid_row_points, crs)
    # y value changes across the row indices
    row_indices = [list(x_y_values1.keys())[0][0], list(x_y_values1.keys())[1][0]]
    y_values = [list(x_y_values1.values())[0][1], list(x_y_values1.values())[1][1]]

    x_y_values2 = get_x_y_values_from_geographic_points(geo_grid_col_points, crs)
    # x value changes across the col indices
    col_indices = [list(x_y_values2.keys())[0][1], list(x_y_values2.keys())[1][1]]
    x_values = [list(x_y_values2.values())[0][0], list(x_y_values2.values())[1][0]]

    y_dim = get_dim_array_data_from_dimvalues(y_values, row_indices, row_size)
    x_dim = get_dim_array_data_from_dimvalues(x_values, col_indices, col_size)

    projected_y, projected_x = tuple(projected_dimension_names)
    return {projected_y: y_dim, projected_x: x_dim}
