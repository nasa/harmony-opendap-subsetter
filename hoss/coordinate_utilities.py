""" This module contains utility functions used for
    coordinate variables and methods to convert the
    coordinate variable data to x/y dimension scales
"""

from typing import Dict, Set, Tuple

import numpy as np
from netCDF4 import Dataset
from numpy import ndarray
from pyproj import CRS, Transformer
from varinfo import VariableFromDmr, VarInfoFromDmr

from hoss.exceptions import (
    CannotComputeDimensionResolution,
    InvalidCoordinateVariable,
    IrregularCoordinateVariables,
    MissingCoordinateVariable,
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

    lat_fill, lon_fill = get_fill_values_for_coordinates(
        latitude_coordinate, longitude_coordinate
    )

    row_size, col_size = get_row_col_sizes_from_coordinate_datasets(
        lat_arr,
        lon_arr,
    )
    geo_grid_points = get_two_valid_geo_grid_points(
        lat_arr, lon_arr, lat_fill, lon_fill, row_size, col_size
    )

    x_y_values = get_x_y_values_from_geographic_points(geo_grid_points, crs)

    row_indices, col_indices = zip(*list(x_y_values.keys()))

    x_values, y_values = zip(*list(x_y_values.values()))

    y_dim = get_dimension_scale_from_dimvalues(y_values, row_indices, row_size)

    x_dim = get_dimension_scale_from_dimvalues(x_values, col_indices, col_size)

    return {'projected_y': y_dim, 'projected_x': x_dim}


def get_row_col_sizes_from_coordinate_datasets(
    lat_arr: ndarray,
    lon_arr: ndarray,
) -> Tuple[int, int]:
    """
    This method returns the row and column sizes of the coordinate datasets

    """
    row_size = 0
    col_size = 0
    if lat_arr.ndim > 1 and lon_arr.shape == lat_arr.shape:
        col_size = lat_arr.shape[0]
        row_size = lat_arr.shape[1]
    elif lat_arr.ndim == 1 and lon_arr.ndim == 1:
        col_size = lat_arr.size
        row_size = lon_arr.size
    elif lon_arr.shape != lat_arr.shape:
        raise IrregularCoordinateVariables(lon_arr.ndim, lat_arr.ndim)
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
        lon_arr = prefetch_dataset[longitude_coordinate.full_name_path][:]
        return lat_arr, lon_arr
    except Exception as exception:
        raise MissingCoordinateVariable('latitude/longitude') from exception


def get_two_valid_geo_grid_points(
    lat_arr: ndarray,
    lon_arr: ndarray,
    lat_fill: float,
    lon_fill: float,
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
    next_col_row_index = -1
    next_col_col_index = 1
    lat_row_valid_indices = lon_row_valid_indices = np.empty((0, 0))

    # get the first row with points that are valid in the lat and lon rows
    first_row_row_index, lat_row_valid_indices = get_valid_indices_in_dataset(
        lat_arr, row_size, lat_fill, 'latitude', 'row', first_row_row_index
    )
    first_row_row_index1, lon_row_valid_indices = get_valid_indices_in_dataset(
        lon_arr, row_size, lon_fill, 'longitude', 'row', first_row_row_index
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

    # get a valid column from the latitude and longitude datasets
    next_col_col_index, lon_col_valid_indices = get_valid_indices_in_dataset(
        lon_arr, col_size, lon_fill, 'longitude', 'col', next_col_col_index
    )
    next_col_col_index1, lat_col_valid_indices = get_valid_indices_in_dataset(
        lat_arr, col_size, lat_fill, 'latitude', 'col', next_col_col_index
    )

    # get a point that is common to both column datasets
    if (
        (next_col_col_index == next_col_col_index1)
        and (lat_col_valid_indices.size > 0)
        and (lon_col_valid_indices.size > 0)
    ):
        next_col_row_index = np.intersect1d(
            lat_col_valid_indices, lon_col_valid_indices
        )[-1]

    # if the whole row and whole column has no valid indices
    # we throw an exception now. This can be extended to move
    # to the next row/col
    if first_row_col_index == -1:
        raise InvalidCoordinateVariable('latitude/longitude')
    if next_col_row_index == -1:
        raise InvalidCoordinateVariable('latitude/longitude')

    geo_grid_indexes = [
        (first_row_row_index, first_row_col_index),
        (next_col_row_index, next_col_col_index),
    ]

    geo_grid_points = [
        (
            lon_arr[first_row_row_index][first_row_col_index],
            lat_arr[first_row_row_index][first_row_col_index],
        ),
        (
            lon_arr[next_col_row_index][next_col_col_index],
            lat_arr[next_col_row_index][next_col_col_index],
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
        dim_max = dim_values[0] + (dim_resolution * dim_indices[0])
        dim_min = dim_values[0] - (-dim_resolution * (dim_size - dim_indices[0] - 1))
        dim_data = np.linspace(dim_max, dim_min, dim_size)

    return dim_data


def get_valid_indices_in_dataset(
    coordinate_arr: ndarray,
    dim_size: int,
    coordinate_fill: float,
    coordinate_name: str,
    span_type: str,
    start_index: int,
) -> tuple[int, ndarray]:
    """
    This method gets valid indices in a row or column of a
    coordinate dataset
    """
    coordinate_index = start_index
    valid_indices = []
    if span_type == 'row':
        valid_indices = get_valid_indices(
            coordinate_arr[coordinate_index, :], coordinate_fill, coordinate_name
        )
    else:
        valid_indices = get_valid_indices(
            coordinate_arr[:, coordinate_index], coordinate_fill, coordinate_name
        )
        while valid_indices.size == 0:
            if coordinate_index < dim_size:
                coordinate_index = coordinate_index + 1
                if span_type == 'row':
                    valid_indices = get_valid_indices(
                        coordinate_arr[coordinate_index, :],
                        coordinate_fill,
                        coordinate_name,
                    )
                else:
                    valid_indices = get_valid_indices(
                        coordinate_arr[:, coordinate_index],
                        coordinate_fill,
                        coordinate_name,
                    )
            else:
                raise InvalidCoordinateVariable(coordinate_name)
    return coordinate_index, valid_indices


def get_valid_indices(
    coordinate_row_col: ndarray, coordinate_fill: float, coordinate_name: str
) -> ndarray:
    """
    Returns indices of a valid array without fill values
    """
    valid_indices = np.empty((0, 0))
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

    return valid_indices


def get_fill_values_for_coordinates(
    latitude_coordinate: VariableFromDmr,
    longitude_coordinate: VariableFromDmr,
) -> tuple[float | None, float | None]:
    """
    returns fill values for the variable. If it does not exist
    checks for the overrides from the json file. If there is no
    overrides, returns None
    """

    lat_fill = None
    lon_fill = None
    lat_fill_value = latitude_coordinate.get_attribute_value('_FillValue')
    lon_fill_value = longitude_coordinate.get_attribute_value('_FillValue')

    if lat_fill_value is not None:
        lat_fill = float(lat_fill_value)
    if lon_fill_value is not None:
        lon_fill = float(lon_fill_value)

    return float(lat_fill), float(lon_fill)
