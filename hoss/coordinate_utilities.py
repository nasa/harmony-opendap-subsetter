""" This module contains utility functions used for
    coordinate variables and methods to convert the
    coordinate variable data to x/y dimension scales
"""

from typing import Dict, Set, Tuple

import numpy as np
from netCDF4 import Dataset
from numpy import ndarray
from pyproj import CRS
from varinfo import VariableFromDmr, VarInfoFromDmr

from hoss.exceptions import (
    IrregularCoordinateDatasets,
    MissingCoordinateDataset,
    MissingValidCoordinateDataset,
)
from hoss.projection_utilities import (
    get_x_y_extents_from_geographic_points,
)


def get_override_projected_dimension_name(
    varinfo: VarInfoFromDmr,
    variable_name: str,
) -> str:
    """returns the x-y projection variable names that would
    match the geo coordinate names. The `latitude` coordinate
    variable name gets converted to 'projected_y' dimension scale
    and the `longitude` coordinate variable name gets converted to
    'projected_x'

    """
    override_variable = varinfo.get_variable(variable_name)
    projected_dimension_name = ''
    if override_variable is not None:
        if override_variable.is_latitude():
            projected_dimension_name = 'projected_y'
        elif override_variable.is_longitude():
            projected_dimension_name = 'projected_x'
    return projected_dimension_name


def get_override_projected_dimensions(
    varinfo: VarInfoFromDmr,
    variable_name: str,
) -> list[str]:
    """
    Returns the projected dimensions names from coordinate variables
    """
    latitude_coordinates, longitude_coordinates = get_coordinate_variables(
        varinfo, [variable_name]
    )
    if latitude_coordinates and longitude_coordinates:
        # there should be only 1 lat and lon coordinate for one variable
        override_dimensions = []
        override_dimensions.append(
            get_override_projected_dimension_name(varinfo, latitude_coordinates[0])
        )
        override_dimensions.append(
            get_override_projected_dimension_name(varinfo, longitude_coordinates[0])
        )

    else:
        # if the override is the variable
        override_projected_dimension_name = get_override_projected_dimension_name(
            varinfo, variable_name
        )
        override_dimensions = ['projected_y', 'projected_x']
        if override_projected_dimension_name not in override_dimensions:
            override_dimensions = []
    return override_dimensions


def get_variables_with_anonymous_dims(
    varinfo: VarInfoFromDmr, required_variables: set[str]
) -> bool:
    """
    returns the list of required variables without any
    dimensions
    """
    return set(
        required_variable
        for required_variable in required_variables
        if len(varinfo.get_variable(required_variable).dimensions) == 0
    )


def get_coordinate_variables(
    varinfo: VarInfoFromDmr,
    requested_variables: Set[str],
) -> tuple[list, list]:
    """This method returns coordinate variables that are referenced
    in the variables requested. It returns it in a specific order
    [latitude, longitude]
    """

    coordinate_variables_set = varinfo.get_references_for_attribute(
        requested_variables, 'coordinates'
    )

    latitude_coordinate_variables = [
        coordinate
        for coordinate in coordinate_variables_set
        if varinfo.get_variable(coordinate).is_latitude()
    ]

    longitude_coordinate_variables = [
        coordinate
        for coordinate in coordinate_variables_set
        if varinfo.get_variable(coordinate).is_longitude()
    ]

    return latitude_coordinate_variables, longitude_coordinate_variables


def update_dimension_variables(
    prefetch_dataset: Dataset,
    latitude_coordinate: VariableFromDmr,
    longitude_coordinate: VariableFromDmr,
    crs: CRS,
) -> Dict[str, ndarray]:
    """Generate artificial 1D dimensions variable for each
    2D dimension or coordinate variable

    For each dimension variable:
    (1) Check if the dimension variable is 1D.
    (2) If it is not 1D and is 2D get the dimension sizes
    (3) Get the corner points from the coordinate variables
    (4) Get the x-y max-min values
    (5) Generate the x-y dimscale array and return to the calling method

    """
    lat_arr, lon_arr = get_lat_lon_arrays(
        prefetch_dataset,
        latitude_coordinate,
        longitude_coordinate,
    )
    if not lat_arr.size:
        raise MissingCoordinateDataset('latitude')
    if not lon_arr.size:
        raise MissingCoordinateDataset('longitude')

    lat_fill, lon_fill = get_fill_values_for_coordinates(
        latitude_coordinate, longitude_coordinate
    )

    row_size, col_size = get_row_col_sizes_from_coordinate_datasets(
        lat_arr,
        lon_arr,
    )

    geo_grid_corners = get_geo_grid_corners(
        lat_arr,
        lon_arr,
        lat_fill,
        lon_fill,
    )

    x_y_extents = get_x_y_extents_from_geographic_points(geo_grid_corners, crs)

    # get grid size and resolution
    x_min = x_y_extents['x_min']
    x_max = x_y_extents['x_max']
    y_min = x_y_extents['y_min']
    y_max = x_y_extents['y_max']
    x_resolution = (x_max - x_min) / row_size
    y_resolution = (y_max - y_min) / col_size

    # create the xy dim scales
    lat_asc, lon_asc = is_lat_lon_ascending(lat_arr, lon_arr, lat_fill, lon_fill)

    if lon_asc:
        x_dim = np.arange(x_min, x_max, x_resolution)
    else:
        x_dim = np.arange(x_min, x_max, -x_resolution)

    if lat_asc:
        y_dim = np.arange(y_max, y_min, y_resolution)
    else:
        y_dim = np.arange(y_max, y_min, -y_resolution)

    return {'projected_y': y_dim, 'projected_x': x_dim}


def get_row_col_sizes_from_coordinate_datasets(
    lat_arr: ndarray,
    lon_arr: ndarray,
) -> Tuple[int, int]:
    """
    This method returns the row and column sizes of the coordinate datasets

    """

    if lat_arr.ndim > 1:
        col_size = lat_arr.shape[0]
        row_size = lat_arr.shape[1]
    if (lon_arr.shape[0] != lat_arr.shape[0]) or (lon_arr.shape[1] != lat_arr.shape[1]):
        raise IrregularCoordinateDatasets(lon_arr.shape, lat_arr.shape)
    if lat_arr.ndim and lon_arr.ndim == 1:
        col_size = lat_arr.size
        row_size = lon_arr.size
    return row_size, col_size


def is_lat_lon_ascending(
    lat_arr: ndarray,
    lon_arr: ndarray,
    lat_fill: float,
    lon_fill: float,
) -> tuple[bool, bool]:
    """
    Checks if the latitude and longitude cooordinate datasets have values
    that are ascending
    """

    lat_col = lat_arr[:, 0]
    lon_row = lon_arr[0, :]

    lat_col_valid_indices = get_valid_indices(lon_row, lat_fill, 'latitude')
    latitude_ascending = (
        lat_col[lat_col_valid_indices[1]] > lat_col[lat_col_valid_indices[0]]
    )

    lon_row_valid_indices = get_valid_indices(lon_row, lon_fill, 'longitude')
    longitude_ascending = (
        lon_row[lon_row_valid_indices[1]] > lon_row[lon_row_valid_indices[0]]
    )

    return latitude_ascending, longitude_ascending


def get_lat_lon_arrays(
    prefetch_dataset: Dataset,
    latitude_coordinate: VariableFromDmr,
    longitude_coordinate: VariableFromDmr,
) -> Tuple[ndarray, ndarray]:
    """
    This method is used to return the lat lon arrays from a 2D
    coordinate dataset.
    """
    lat_arr = []
    lon_arr = []

    lat_arr = prefetch_dataset[latitude_coordinate.full_name_path][:]
    lon_arr = prefetch_dataset[longitude_coordinate.full_name_path][:]

    return lat_arr, lon_arr


def get_geo_grid_corners(
    lat_arr: ndarray,
    lon_arr: ndarray,
    lat_fill: float,
    lon_fill: float,
) -> list[Tuple[float]]:
    """
    This method is used to return the lat lon corners from a 2D
    coordinate dataset. It gets the row and column of the latitude and longitude
    arrays to get the corner points. This does a check for fill values and
    This method does not check if there are fill values in the corner points
    to go down to the next row and col. The fill values in the corner points
    still needs to be addressed. It will raise an exception in those
    cases.
    """

    top_left_row_idx = 0
    top_left_col_idx = 0

    # get the first row from the longitude dataset
    lon_row = lon_arr[top_left_row_idx, :]
    lon_row_valid_indices = get_valid_indices(lon_row, lon_fill, 'longitude')

    # get the index of the minimum longitude after checking for invalid entries
    top_left_col_idx = lon_row_valid_indices[lon_row[lon_row_valid_indices].argmin()]
    min_lon = lon_row[top_left_col_idx]

    # get the index of the maximum longitude after checking for invalid entries
    top_right_col_idx = lon_row_valid_indices[lon_row[lon_row_valid_indices].argmax()]
    max_lon = lon_row[top_right_col_idx]

    # get the last valid longitude column to get the latitude array
    lat_col = lat_arr[:, top_right_col_idx]
    lat_col_valid_indices = get_valid_indices(lat_col, lat_fill, 'latitude')

    # get the index of minimum latitude after checking for valid values
    bottom_right_row_idx = lat_col_valid_indices[
        lat_col[lat_col_valid_indices].argmin()
    ]
    min_lat = lat_col[bottom_right_row_idx]

    # get the index of maximum latitude after checking for valid values
    top_right_row_idx = lat_col_valid_indices[lat_col[lat_col_valid_indices].argmax()]
    max_lat = lat_col[top_right_row_idx]

    geo_grid_corners = [
        [min_lon, max_lat],
        [max_lon, max_lat],
        [max_lon, min_lat],
        [min_lon, min_lat],
    ]
    return geo_grid_corners


def get_valid_indices(
    coordinate_row_col: ndarray, coordinate_fill: float, coordinate_name: str
) -> ndarray:
    """
    Returns indices of a valid array without fill values
    """
    if coordinate_fill:
        valid_indices = np.where(coordinate_row_col != coordinate_fill)[0]
    elif coordinate_name == 'longitude':
        valid_indices = np.where(
            (coordinate_row_col >= -180.0) & (coordinate_row_col <= 180.0)
        )[0]
    elif coordinate_name == 'latitude':
        valid_indices = np.where(
            (coordinate_row_col >= -90.0) & (coordinate_row_col <= 90.0)
        )[0]

    # if the first row does not have valid indices,
    # should go down to the next row. We throw an exception
    # for now till that gets addressed
    if not valid_indices.size:
        raise MissingValidCoordinateDataset(coordinate_name)
    return valid_indices


def get_fill_values_for_coordinates(
    latitude_coordinate: VariableFromDmr,
    longitude_coordinate: VariableFromDmr,
) -> float | None:
    """
    returns fill values for the variable. If it does not exist
    checks for the overrides from the json file. If there is no
    overrides, returns None
    """

    lat_fill_value = latitude_coordinate.get_attribute_value('_fillValue')
    lon_fill_value = longitude_coordinate.get_attribute_value('_fillValue')
    # if fill_value is None:
    # check if there are overrides in hoss_config.json using varinfo
    # else
    return lat_fill_value, lon_fill_value
