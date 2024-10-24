""" This module contains utility functions used for
    coordinate variables and functions to convert the
    coordinate variable data to projected x/y dimension values
"""

import numpy as np
from netCDF4 import Dataset

# from numpy import ndarray
from varinfo import VariableFromDmr, VarInfoFromDmr

from hoss.exceptions import (
    InvalidCoordinateDataset,
    InvalidCoordinateVariable,
    InvalidSizingInCoordinateVariables,
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
        if len(varinfo.get_variable(variable).dimensions) == 0
    )


def get_coordinate_variables(
    varinfo: VarInfoFromDmr,
    requested_variables: list,
) -> tuple[list, list]:
    """This function returns latitude and longitude variables listed in the
    CF-Convention coordinates metadata attribute. It returns them in a specific
    order [latitude, longitude]"
    """

    coordinate_variables_list = varinfo.get_references_for_attribute(
        requested_variables, 'coordinates'
    )
    latitude_coordinate_variables = [
        coordinate
        for coordinate in coordinate_variables_list
        if varinfo.get_variable(coordinate).is_latitude()
    ]

    longitude_coordinate_variables = [
        coordinate
        for coordinate in coordinate_variables_list
        if varinfo.get_variable(coordinate).is_longitude()
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
        raise InvalidSizingInCoordinateVariables(lon_arr.shape, lat_arr.shape)
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

    # return dim_data


def get_valid_indices(
    coordinate_row_col: np.ndarray, coordinate_fill: float, coordinate_name: str
) -> np.ndarray:
    """
    Returns indices of a valid array without fill values if the fill
    value is provided. If it is not provided, we check for valid values
    for latitude and longitude
    """

    if coordinate_fill:
        valid_indices = np.where(
            ~np.isclose(coordinate_row_col, float(coordinate_fill))
        )[0]
    if coordinate_name == 'longitude':
        valid_indices = np.where(
            # (-180.0 <= coordinate_row_col <= 360.0).all()
            # (coordinate_row_col > -90.0) and (coordinate_row_col < 90.0)
            (coordinate_row_col > -180.0)
            & (coordinate_row_col < 360.0)
        )[0]
    elif coordinate_name == 'latitude':
        valid_indices = np.where(
            # (-90.0 <= coordinate_row_col <= 90.0).all()
            # (coordinate_row_col > -90.0) and (coordinate_row_col < 90.0)
            (coordinate_row_col > -90.0)
            & (coordinate_row_col < 90.0)
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
