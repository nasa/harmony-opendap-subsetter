""" This module contains utility functions used for
    coordinate variables and methods to convert the
    coordinate variable data to x/y dimension scales
"""

from typing import Set, Tuple

import numpy as np
from netCDF4 import Dataset
from numpy import ndarray
from varinfo import VariableFromDmr, VarInfoFromDmr

from hoss.exceptions import (
    CannotComputeDimensionResolution,
    IrregularCoordinateVariables,
    MissingCoordinateVariable,
)


def get_override_projected_dimension_names(
    varinfo: VarInfoFromDmr, variable_name: str
) -> str:
    """returns the x-y projection variable names that would
    match the group of geo coordinate names. The  coordinate
    variable name gets converted to 'projected_y' dimension scale
    and 'projected_x'

    """
    override_variable = varinfo.get_variable(variable_name)

    if override_variable is not None and (
        override_variable.is_latitude() or override_variable.is_longitude()
    ):
        projected_dimension_names = [
            f'{override_variable.group_path}/projected_y',
            f'{override_variable.group_path}/projected_x',
        ]
    else:
        raise MissingCoordinateVariable(override_variable.full_name_path)

    return projected_dimension_names


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

    override_dimensions = []
    if latitude_coordinates and longitude_coordinates:
        # there should be only 1 lat and lon coordinate for one variable
        override_dimensions = get_override_projected_dimension_names(
            varinfo, latitude_coordinates[0]
        )

    # if the override is the variable
    elif (
        varinfo.get_variable(variable_name).is_latitude()
        or varinfo.get_variable(variable_name).is_longitude()
    ):
        override_dimensions = get_override_projected_dimension_names(
            varinfo, variable_name
        )
    return override_dimensions


def get_variables_with_anonymous_dims(
    varinfo: VarInfoFromDmr, variables: set[str]
) -> set[str]:
    """
    returns a set of variables without any
    dimensions
    """
    return set(
        variable
        for variable in variables
        if len(varinfo.get_variable(variable).dimensions) == 0
    )


def get_coordinate_variables(
    varinfo: VarInfoFromDmr,
    requested_variables: Set[str],
) -> tuple[list, list]:
    """This method returns coordinate variables that are referenced
    in the variables requested. It returns it in a specific order
    [latitude, longitude]
    """

    coordinate_variables_set = sorted(
        varinfo.get_references_for_attribute(requested_variables, 'coordinates')
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


def get_row_col_sizes_from_coordinate_datasets(
    lat_arr: ndarray,
    lon_arr: ndarray,
) -> Tuple[int, int]:
    """
    This method returns the row and column sizes of the coordinate datasets

    """
    if lat_arr.ndim > 1 and lon_arr.shape == lat_arr.shape:
        col_size = lat_arr.shape[1]
        row_size = lat_arr.shape[0]
    elif (
        lat_arr.ndim == 1
        and lon_arr.ndim == 1
        and lat_arr.size > 0
        and lon_arr.size > 0
    ):
        col_size = lon_arr.size
        row_size = lat_arr.size
    else:
        raise IrregularCoordinateVariables(lon_arr.shape, lat_arr.shape)
    return row_size, col_size


def get_lat_lon_arrays(
    prefetch_dataset: Dataset,
    latitude_coordinate: VariableFromDmr,
    longitude_coordinate: VariableFromDmr,
) -> Tuple[ndarray, ndarray]:
    """
    This method is used to return the lat lon arrays from a 2D
    coordinate dataset.
    """
    try:
        lat_arr = prefetch_dataset[latitude_coordinate.full_name_path][:]
    except Exception as exception:
        raise MissingCoordinateVariable(
            latitude_coordinate.full_name_path
        ) from exception

    try:
        lon_arr = prefetch_dataset[longitude_coordinate.full_name_path][:]
    except Exception as exception:
        raise MissingCoordinateVariable(
            longitude_coordinate.full_name_path
        ) from exception

    return lat_arr, lon_arr


def get_dimension_scale_from_dimvalues(
    dim_values: ndarray, dim_indices: ndarray, dim_size: float
) -> ndarray:
    """
    return a full dimension scale based on the 2 projected points and
    grid size
    """
    dim_resolution = 0.0
    if (dim_indices[1] != dim_indices[0]) and (dim_values[1] != dim_values[0]):
        dim_resolution = (dim_values[1] - dim_values[0]) / (
            dim_indices[1] - dim_indices[0]
        )
    if dim_resolution == 0.0:
        raise CannotComputeDimensionResolution(dim_values[0], dim_indices[0])

    # create the dim scale
    dim_asc = dim_values[1] > dim_values[0]

    if dim_asc:
        dim_min = dim_values[0] + (dim_resolution * dim_indices[0])
        dim_max = dim_values[0] + (dim_resolution * (dim_size - dim_indices[0] - 1))
        dim_data = np.linspace(dim_min, dim_max, dim_size)
    else:
        dim_max = dim_values[0] + (-dim_resolution * dim_indices[0])
        dim_min = dim_values[0] - (-dim_resolution * (dim_size - dim_indices[0] - 1))
        dim_data = np.linspace(dim_max, dim_min, dim_size)

    return dim_data


def get_valid_indices(
    coordinate_row_col: ndarray, coordinate_fill: float, coordinate_name: str
) -> ndarray:
    """
    Returns indices of a valid array without fill values
    """

    if coordinate_fill:
        valid_indices = np.where(
            ~np.isclose(coordinate_row_col, float(coordinate_fill))
        )[0]
    elif coordinate_name == 'longitude':
        valid_indices = np.where(
            (coordinate_row_col >= -180.0) & (coordinate_row_col <= 180.0)
        )[0]
    elif coordinate_name == 'latitude':
        valid_indices = np.where(
            (coordinate_row_col >= -90.0) & (coordinate_row_col <= 90.0)
        )[0]
    else:
        valid_indices = np.empty((0, 0))

    return valid_indices


def get_fill_value_for_coordinate(
    coordinate: VariableFromDmr,
) -> float | None:
    """
    returns fill values for the variable. If it does not exist
    checks for the overrides from the json file. If there is no
    overrides, returns None
    """

    fill_value = coordinate.get_attribute_value('_FillValue')
    if fill_value is not None:
        return float(fill_value)
    return fill_value
