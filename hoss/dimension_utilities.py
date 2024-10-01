""" This module contains utility functions for dimension variables, including
    the conversion from user-defined extents to array indices, and filling of
    variables if the requested range crosses a grid edge.

    These utilities are compatible with any type of dimension. If specific
    handling is required for the extents or dimension values themselves, that
    handling should occur prior to calling functions from this module. For
    example, ensuring requested bounding box longitudes are wrapped or
    unwrapped in accordance with the longitude dimension values.

"""

from logging import Logger
from pathlib import PurePosixPath
from typing import Dict, Set, Tuple

import numpy as np
from harmony.message import Message
from harmony.message_utility import rgetattr
from harmony.util import Config
from netCDF4 import Dataset
from numpy import ndarray
from numpy.ma.core import MaskedArray
from pyproj import CRS
from varinfo import VariableFromDmr, VarInfoFromDmr

from hoss.exceptions import (
    InvalidNamedDimension,
    InvalidRequestedRange,
    IrregularCoordinateDatasets,
    MissingCoordinateDataset,
    MissingValidCoordinateDataset,
)
from hoss.projection_utilities import (
    get_x_y_extents_from_geographic_points,
)
from hoss.utilities import (
    format_variable_set_string,
    get_opendap_nc4,
    get_value_or_default,
)

IndexRange = Tuple[int]
IndexRanges = Dict[str, IndexRange]


def is_index_subset(message: Message) -> bool:
    """Determine if the inbound Harmony request specified any parameters that
    will require an index range subset. These will be:

    * Bounding box spatial requests (Message.subset.bbox)
    * Shape file spatial requests (Message.subset.shape)
    * Temporal requests (Message.temporal)
    * Named dimension range subsetting requests (Message.subset.dimensions)

    """
    return any(
        rgetattr(message, subset_parameter, None) not in (None, [])
        for subset_parameter in [
            'subset.bbox',
            'subset.shape',
            'subset.dimensions',
            'temporal',
        ]
    )


def get_prefetch_variables(
    opendap_url: str,
    varinfo: VarInfoFromDmr,
    required_variables: Set[str],
    output_dir: str,
    logger: Logger,
    access_token: str,
    config: Config,
) -> tuple[str, set[str]] | str:
    """Determine the dimensions that need to be "pre-fetched" from OPeNDAP in
    order to derive index ranges upon them. Initially, this was just
    spatial and temporal dimensions, but to support generic dimension
    subsets, all required dimensions must be prefetched, along with any
    associated bounds variables referred to via the "bounds" metadata
    attribute. In cases where dimension variables do not exist, coordinate
    variables will be prefetched and used to calculate dimension-scale values

    """
    required_dimensions = varinfo.get_required_dimensions(required_variables)
    if required_dimensions:
        bounds = varinfo.get_references_for_attribute(required_dimensions, 'bounds')
        required_dimensions.update(bounds)
    else:
        coordinate_variables = get_coordinate_variables(varinfo, required_variables)
        if coordinate_variables:
            required_dimensions = set(coordinate_variables)

    logger.info(
        'Variables being retrieved in prefetch request: '
        f'{format_variable_set_string(required_dimensions)}'
    )

    required_dimensions_nc4 = get_opendap_nc4(
        opendap_url, required_dimensions, output_dir, logger, access_token, config
    )

    # Create bounds variables if necessary.
    add_bounds_variables(required_dimensions_nc4, required_dimensions, varinfo, logger)
    return required_dimensions_nc4


def get_override_projected_dimensions(
    varinfo: VarInfoFromDmr,
    override_variable_name: str,
) -> str | None:
    """returns the x-y projection variable names that would
    match the geo coordinate names. The `latitude` coordinate
    variable name gets converted to 'projected_y' dimension scale
    and the `longitude` coordinate variable name gets converted to
    'projected_x'

    """
    override_variable = varinfo.get_variable(override_variable_name)
    if override_variable is not None:
        if override_variable.is_latitude():
            projected_dimension_name = 'projected_y'
        elif override_variable.is_longitude():
            projected_dimension_name = 'projected_x'
        else:
            projected_dimension_name = None
    return projected_dimension_name


def get_coordinate_variables(
    varinfo: VarInfoFromDmr,
    requested_variables: Set[str],
) -> list[str]:
    """This method returns coordinate variables that are referenced
    in the variables requested. It returns it in a specific order
    [latitude, longitude]
    """

    # try:
    coordinate_variables_set = varinfo.get_references_for_attribute(
        requested_variables, 'coordinates'
    )
    coordinate_variables = []
    for coordinate in coordinate_variables_set:
        if varinfo.get_variable(coordinate).is_latitude():
            coordinate_variables.insert(0, coordinate)
        elif varinfo.get_variable(coordinate).is_longitude():
            coordinate_variables.insert(1, coordinate)

    return coordinate_variables
    # except AttributeError:
    #    return set()


def update_dimension_variables(
    prefetch_dataset: Dataset, coordinates: Set[str], varinfo: VarInfoFromDmr, crs: CRS
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
    row_size, col_size = get_row_col_sizes_from_coordinate_datasets(
        prefetch_dataset, coordinates, varinfo
    )

    geo_grid_corners = get_geo_grid_corners(prefetch_dataset, coordinates, varinfo)

    x_y_extents = get_x_y_extents_from_geographic_points(geo_grid_corners, crs)

    # get grid size and resolution
    x_min = x_y_extents['x_min']
    x_max = x_y_extents['x_max']
    y_min = x_y_extents['y_min']
    y_max = x_y_extents['y_max']
    x_resolution = (x_max - x_min) / row_size
    y_resolution = (y_max - y_min) / col_size

    # create the xy dim scales
    lat_asc, lon_asc = is_lat_lon_ascending(prefetch_dataset, coordinates, varinfo)

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
    prefetch_dataset: Dataset,
    coordinates: Set[str],
    varinfo: VarInfoFromDmr,
) -> Tuple[int, int]:
    """
    This method returns the row and column sizes of the coordinate datasets

    """
    lat_arr, lon_arr = get_lat_lon_arrays(prefetch_dataset, coordinates, varinfo)
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
    prefetch_dataset: Dataset,
    coordinates: Set[str],
    varinfo: VarInfoFromDmr,
) -> tuple[bool, bool]:
    """
    Checks if the latitude and longitude cooordinate datasets have values
    that are ascending
    """
    lat_arr, lon_arr = get_lat_lon_arrays(prefetch_dataset, coordinates, varinfo)
    lat_col = lat_arr[:, 0]
    lon_row = lon_arr[0, :]
    return is_dimension_ascending(lat_col), is_dimension_ascending(lon_row)


def get_lat_lon_arrays(
    prefetch_dataset: Dataset,
    coordinates: Set[str],
    varinfo: VarInfoFromDmr,
) -> Tuple[ndarray, ndarray]:
    """
    This method is used to return the lat lon arrays from a 2D
    coordinate dataset.
    """
    lat_arr = []
    lon_arr = []
    for coordinate in coordinates:
        coordinate_variable = varinfo.get_variable(coordinate)
        if coordinate_variable.is_latitude():
            lat_arr = prefetch_dataset[coordinate_variable.full_name_path][:]
        elif coordinate_variable.is_longitude():
            lon_arr = prefetch_dataset[coordinate_variable.full_name_path][:]

    return lat_arr, lon_arr


def get_geo_grid_corners(
    prefetch_dataset: Dataset,
    coordinates: Set[str],
    varinfo: VarInfoFromDmr,
) -> list[Tuple[float]]:
    """
    This method is used to return the lat lon corners from a 2D
    coordinate dataset. It gets the row and column of the latitude and longitude
    arrays to get the corner points. This does a check for values below -180
    which could be fill values. This method does not check if there
    are fill values in the corner points to go down to the next row and col
    The fill values in the corner points still needs to be addressed.
    """
    lat_arr, lon_arr = get_lat_lon_arrays(prefetch_dataset, coordinates, varinfo)
    if not lat_arr.size:
        raise MissingCoordinateDataset('latitude')
    if not lon_arr.size:
        raise MissingCoordinateDataset('longitude')

    # skip fill values when calculating min values
    # topleft =  minlon, maxlat
    # bottomright = maxlon, minlat
    top_left_row_idx = 0
    top_left_col_idx = 0

    # get the first row from the longitude dataset
    lon_row = lon_arr[top_left_row_idx, :]
    lon_row_valid_indices = np.where((lon_row >= -180.0) & (lon_row <= 180.0))[0]

    # if the first row does not have valid indices,
    # should go down to the next row. We throw an exception
    # for now till that gets addressed
    if not lon_row_valid_indices.size:
        raise MissingValidCoordinateDataset('longitude')

    # get the index of the minimum longitude after checking for invalid entries
    top_left_col_idx = lon_row_valid_indices[lon_row[lon_row_valid_indices].argmin()]
    min_lon = lon_row[top_left_col_idx]

    # get the index of the maximum longitude after checking for invalid entries
    top_right_col_idx = lon_row_valid_indices[lon_row[lon_row_valid_indices].argmax()]
    max_lon = lon_row[top_right_col_idx]

    # get the last valid longitude column to get the latitude array
    lat_col = lat_arr[:, top_right_col_idx]
    lat_col_valid_indices = np.where((lat_col >= -90.0) & (lat_col <= 90.0))[0]

    # if the longitude values are invalid, should check the other columns
    # We throw an exception for now till that gets addressed
    if not lat_col_valid_indices.size:
        raise MissingValidCoordinateDataset('latitude')

    # get the index of minimum latitude after checking for valid values
    bottom_right_row_idx = lat_col_valid_indices[
        lat_col[lat_col_valid_indices].argmin()
    ]
    min_lat = lat_col[bottom_right_row_idx]

    # get the index of maximum latitude after checking for valid values
    top_right_row_idx = lat_col_valid_indices[lat_col[lat_col_valid_indices].argmax()]
    max_lat = lat_col[top_right_row_idx]

    top_left_corner = [min_lon, max_lat]
    top_right_corner = [max_lon, max_lat]
    bottom_right_corner = [max_lon, min_lat]
    bottom_left_corner = [min_lon, min_lat]

    geo_grid_corners = [
        top_left_corner,
        top_right_corner,
        bottom_right_corner,
        bottom_left_corner,
    ]
    return geo_grid_corners


def add_bounds_variables(
    dimensions_nc4: str,
    required_dimensions: Set[str],
    varinfo: VarInfoFromDmr,
    logger: Logger,
) -> None:
    """Augment a NetCDF4 file with artificial bounds variables for each
    dimension variable that has been identified by the earthdata-varinfo
    configuration file to have an edge-aligned attribute"

    For each dimension variable:
    (1) Check if the variable needs a bounds variable.
    (2) If so, create a bounds array from within the `write_bounds`
        function.
    (3) Then write the bounds variable to the NetCDF4 URL.

    """
    with Dataset(dimensions_nc4, 'r+') as prefetch_dataset:
        for dimension_name in required_dimensions:
            dimension_variable = varinfo.get_variable(dimension_name)
            if needs_bounds(dimension_variable):
                write_bounds(prefetch_dataset, dimension_variable)

                logger.info(
                    'Artificial bounds added for dimension variable: '
                    f'{dimension_name}'
                )


def needs_bounds(dimension: VariableFromDmr) -> bool:
    """Check if a dimension variable needs a bounds variable.
    This will be the case when dimension cells are edge-aligned
    and bounds for that dimension do not already exist.

    """
    return (
        dimension.attributes.get('cell_alignment') == 'edge'
        and dimension.references.get('bounds') is None
    )


def get_bounds_array(prefetch_dataset: Dataset, dimension_path: str) -> np.ndarray:
    """Create an array containing the minimum and maximum bounds
    for each pixel in a given dimension.

    The minimum and maximum values are determined under the assumption
    that the dimension data is monotonically increasing and contiguous.
    So for every bounds but the last, the bounds are simply extracted
    from the dimension dataset.

    The final bounds must be calculated with the assumption that
    the last data cell is edge-aligned and thus has a value the does
    not account for the cell length. So, the final bound is determined
    by taking the median of all the resolutions in the dataset to obtain
    a resolution that can be added to the final data value.

    Ex: Input dataset with resolution of 3 degrees:  [ ... , 81, 84, 87]

    Minimum | Maximum
     <...>     <...>
      81        84
      84        87
      87        ?  ->  87 + median resolution -> 87 + 3 -> 90

    """
    # Access the dimension variable's data using the variable's full path.
    dimension_array = prefetch_dataset[dimension_path][:]

    median_resolution = np.median(np.diff(dimension_array))

    # This array is the transpose of what is required, just for easier assignment
    # of values (indices are [row, column]) during the bounds calculations:
    cell_bounds = np.zeros(shape=(2, dimension_array.size), dtype=dimension_array.dtype)

    # Minimum values are equal to the dimension pixel values (for lower left pixel alignment):
    cell_bounds[0] = dimension_array[:]

    # Maximum values are the next dimension pixel values (for lower left pixel alignment),
    # so these values almost mirror the minimum values but start at the second pixel
    # instead of the first. Here we calculate each bound except for the very last one.
    cell_bounds[1][:-1] = dimension_array[1:]

    # Last maximum value is the last pixel value (minimum) plus the median resolution:
    cell_bounds[1][-1] = dimension_array[-1] + median_resolution

    # Return transpose of array to get correct shape:
    return cell_bounds.T


def write_bounds(
    prefetch_dataset: Dataset, dimension_variable: VariableFromDmr
) -> None:
    """Write the input bounds array to a given dimension dataset.

    First a new dimension is created for the new bounds variable
    to allow the variable to be two-dimensional.

    Then the new bounds variable is created using two dimensions:
    (1) the existing dimension of the dimension dataset, and
    (2) the new bounds variable dimension.

    """
    bounds_array = get_bounds_array(prefetch_dataset, dimension_variable.full_name_path)

    # Create the second bounds dimension.
    dimension_name = str(PurePosixPath(dimension_variable.full_name_path).name)
    dimension_group = str(PurePosixPath(dimension_variable.full_name_path).parent)
    bounds_name = dimension_name + '_bnds'
    bounds_full_path_name = dimension_variable.full_name_path + '_bnds'
    bounds_dimension_name = dimension_name + 'v'

    if dimension_group == '/':
        # The root group must be explicitly referenced here.
        bounds_dim = prefetch_dataset.createDimension(bounds_dimension_name, 2)
    else:
        bounds_dim = prefetch_dataset[dimension_group].createDimension(
            bounds_dimension_name, 2
        )

    # Dimension variables only have one dimension - themselves.
    variable_dimension = prefetch_dataset[dimension_variable.full_name_path].dimensions[
        0
    ]

    bounds_data_type = str(dimension_variable.data_type)
    bounds = prefetch_dataset.createVariable(
        bounds_full_path_name,
        bounds_data_type,
        (
            variable_dimension,
            bounds_dim,
        ),
    )

    # Write data to the new variable in the prefetch dataset.
    bounds[:] = bounds_array[:]

    # Update varinfo attributes and references.
    prefetch_dataset[dimension_variable.full_name_path].setncatts(
        {'bounds': bounds_name}
    )
    dimension_variable.references['bounds'] = {
        bounds_name,
    }
    dimension_variable.attributes['bounds'] = bounds_name


def is_dimension_ascending(dimension: MaskedArray) -> bool:
    """Read the array associated with a dimension variable and check if the
    variables ascend starting from the zeroth element or not.

    """
    first_index, last_index = np.ma.flatnotmasked_edges(dimension)
    return dimension.size == 1 or dimension[first_index] < dimension[last_index]


def get_dimension_index_range(
    dimension_values: MaskedArray,
    request_min: float,
    request_max: float,
    bounds_values: MaskedArray = None,
) -> IndexRange:
    """Ensure that both a minimum and maximum value are defined from the
    message, if not, use the first or last value in the dimension array,
    accordingly. For granules that only contain dimension variables (not
    additional bounds variables) the minimum and maximum values must be
    ordered to be ascending or descending in a way that matches the
    dimension index values.

    Once the minimum and maximum values are determined, and sorted in the
    same order as the dimension array values, retrieve the index values
    that correspond to the requested dimension values. Alternatively, if a
    dimension has an associated bounds variable, use this to determine the
    dimension index range.

    """
    if is_dimension_ascending(dimension_values):
        dimension_min = get_value_or_default(request_min, dimension_values[0])
        dimension_max = get_value_or_default(request_max, dimension_values[-1])
    else:
        dimension_min = get_value_or_default(request_max, dimension_values[0])
        dimension_max = get_value_or_default(request_min, dimension_values[-1])

    if bounds_values is None:
        index_range = get_dimension_indices_from_values(
            dimension_values, dimension_min, dimension_max
        )
    else:
        index_range = get_dimension_indices_from_bounds(
            bounds_values,
            min(dimension_min, dimension_max),
            max(dimension_min, dimension_max),
        )

    return index_range


def get_dimension_indices_from_values(
    dimension: MaskedArray, minimum_extent: float, maximum_extent: float
) -> IndexRange:
    """Find the indices closest to the interpolated values of the minimum and
    maximum extents in that dimension.

    Use of `numpy.interp` maps the dimension scale values to their index
    values and then computes an interpolated index value that best matches
    the bounding value (minimum_extent, maximum_extent) to a "fractional"
    index value. Rounding that value gives the starting index value for the
    cell that contains that bound.

    If an extent is requested that is a single point in this dimension, the
    range should be the two surrounding pixels that border the point.

    For an ascending dimension:

    * `minimum_extent` ≤ `maximum_extent`.

    For a descending dimension:

    * `minimum_extent` ≥ `maximum_extent`

    Input longitude extent values must conform to the valid range of the
    native dimension data.

    """
    dimension_range = [minimum_extent, maximum_extent]
    dimension_indices = np.arange(dimension.size)

    if is_dimension_ascending(dimension):
        dimension_values = dimension
    else:
        # second argument to `np.interp` must be ascending.
        # The dimension indices also should be flipped to still be correct
        dimension_values = np.flip(dimension)
        dimension_indices = np.flip(dimension_indices)

    raw_indices = np.interp(dimension_range, dimension_values, dimension_indices)

    if (raw_indices[0] == raw_indices[1]) and (raw_indices[0] % 1 == 0.5):
        # Minimum extent is exactly halfway between two pixels, and the
        # requested extents are a single point in this dimension. Round down
        # to retrieve the two bordering pixels.
        raw_minimum_index = np.nextafter(raw_indices[0], raw_indices[0] - 1)
    elif raw_indices[0] % 1 == 0.5:
        # Minimum extent is exactly halfway between two pixels, round up.
        raw_minimum_index = np.nextafter(raw_indices[0], raw_indices[0] + 1)
    else:
        raw_minimum_index = raw_indices[0]

    if (raw_indices[0] == raw_indices[1]) and (raw_indices[1] % 1 == 0.5):
        # Maximum extent is exactly halfway between two pixels, and the
        # requested extents are a single point in this dimension. Round up to
        # retrieve the two bordering pixels.
        raw_maximum_index = np.nextafter(raw_indices[1], raw_indices[1] + 1)
    elif raw_indices[1] % 1 == 0.5:
        # Maximum extent is exactly halfway between two pixels, round down.
        raw_maximum_index = np.nextafter(raw_indices[1], raw_indices[1] - 1)
    else:
        raw_maximum_index = raw_indices[1]

    minimum_index = int(np.rint(raw_minimum_index))
    maximum_index = int(np.rint(raw_maximum_index))

    return (minimum_index, maximum_index)


def get_dimension_indices_from_bounds(
    bounds: np.ndarray, min_value: float, max_value: float
) -> Tuple[int]:
    """Derive the dimension array indices that correspond to the requested
    dimension range in the input Harmony message.

    This function assumes:

    - The pixels bounds represent a contiguous range, e.g., the upper
      bound of one pixel is always equal to the lower bound of the next
      pixel.
    - The bounds arrays are monotonic in the 0th dimension (e.g., lower and
      upper bounds values either all ascend or all descend along with the
      array indices).
    - min_value ≤ max_value.

    """
    if min_value > np.nanmax(bounds) or max_value < np.nanmin(bounds):
        raise InvalidRequestedRange()

    if is_dimension_ascending(bounds.T[0]):
        # Lower bounds are the first row, upper bounds are the second:
        minimum_index = np.where(bounds.T[1] > min_value)[0][0]
        maximum_index = np.where(bounds.T[0] < max_value)[0][-1]
    else:
        # Lower bounds are the second row, upper bounds are the first:
        minimum_index = np.where(bounds.T[1] < max_value)[0][0]
        maximum_index = np.where(bounds.T[0] > min_value)[0][-1]

    if (min_value == max_value) and is_almost_in(min_value, bounds.T[0][1:]):
        # Single point in dimension that lies exactly on a boundary between
        # two pixels - return indices for both surrounding pixels.
        minimum_index -= 1
        maximum_index += 1

    return (minimum_index, maximum_index)


def add_index_range(
    variable_name: str,
    varinfo: VarInfoFromDmr,
    index_ranges: IndexRanges,
) -> str:
    """Append the index ranges of each dimension for the specified variable.
    If there are no dimensions with listed index ranges, then the full
    variable should be requested, and no index notation is required.
    A variable with a bounding box crossing the edge of the grid (e.g., at
    the antimeridian or Prime Meridian) will have a minimum index greater
    than the maximum index. In this case the full dimension range should be
    requested, as the related values will be masked before returning the
    output to the user.

    """
    variable = varinfo.get_variable(variable_name)
    range_strings = []
    if variable.dimensions:
        range_strings = get_range_strings(variable.dimensions, index_ranges)
    else:
        coordinate_variables = get_coordinate_variables(varinfo, [variable_name])
        if coordinate_variables:
            dimensions = []
            for coordinate in coordinate_variables:
                dimensions.append(
                    get_override_projected_dimensions(varinfo, coordinate)
                )
            range_strings = get_range_strings(dimensions, index_ranges)
        else:
            # if the override is the variable
            override = get_override_projected_dimensions(varinfo, variable_name)
            dimensions = ['projected_y', 'projected_x']
            if override is not None and override in dimensions:
                range_strings = get_range_strings(dimensions, index_ranges)
            else:
                range_strings.append('[]')

    if all(range_string == '[]' for range_string in range_strings):
        indices_string = ''
    else:
        indices_string = ''.join(range_strings)
    return f'{variable_name}{indices_string}'


def get_range_strings(
    variable_dimensions: list,
    index_ranges: IndexRanges,
) -> list:
    """Calculates the index ranges for each dimension of the variable
    and returns the list of index ranges
    """
    range_strings = []
    for dimension in variable_dimensions:
        dimension_range = index_ranges.get(dimension)
        if dimension_range is not None and dimension_range[0] <= dimension_range[1]:
            range_strings.append(f'[{dimension_range[0]}:{dimension_range[1]}]')
        else:
            range_strings.append('[]')

    return range_strings


def get_fill_slice(dimension: str, fill_ranges: IndexRanges) -> slice:
    """Check the dictionary of dimensions that need to be filled for the
    given dimension. If present, the minimum index will be greater than the
    maximum index (the eastern edge of the bounding box will seem to be to
    the west of the western edge due to crossing the grid edge). The region
    to be filled is between these indices:

    * Start index = maximum index + 1.
    * Stop index = minimum index. (As Python index slices go up to, but not
      including, the stop index).

    If the dimension is not to be filled, return a `slice` with unspecified
    start and stop. This is the equivalent of the full range in this
    dimension. Slices for all variable dimensions will be combined to
    identify the region of the variable to be filled, e.g.:

    * variable[(slice(None), slice(None), slice(start_lon, stop_lon))] = fill

    This is equivalent to:

    * science_variable[:][:][start_lon:stop_lon] = fill

    """
    if dimension in fill_ranges:
        fill_slice = slice(fill_ranges[dimension][1] + 1, fill_ranges[dimension][0])
    else:
        fill_slice = slice(None)

    return fill_slice


def get_dimension_extents(dimension_array: np.ndarray) -> Tuple[float]:
    """Fit the dimension with a straight line, and find the outer edge of the
    first and last pixel, assuming the supplied values lie at the centre of
    each pixel.

    """
    dimension_indices = np.arange(dimension_array.size)
    gradient, _ = np.polyfit(dimension_indices, dimension_array, 1)

    min_extent = dimension_array.min() - np.abs(gradient) / 2.0
    max_extent = dimension_array.max() + np.abs(gradient) / 2.0

    return (min_extent, max_extent)


def get_requested_index_ranges(
    required_variables: Set[str],
    varinfo: VarInfoFromDmr,
    dimensions_path: str,
    harmony_message: Message,
) -> IndexRanges:
    """Examines the requested dimension names and ranges and extracts the
    indices that correspond to the specified range of values for each
    dimension that is requested specifically by name.

    When dimensions, such as atmospheric pressure or ocean depth, have
    values that are descending (getting smaller from start to finish), then
    the min/max values of the requested range are flipped. If the dimension
    is descending, the specified range must also be descending.

    The return value from this function is a dictionary that contains the
    index ranges for the named dimension, such as: {'/lev': [1, 5]}

    """
    required_dimensions = varinfo.get_required_dimensions(required_variables)

    dim_index_ranges = {}

    with Dataset(dimensions_path, 'r') as dimensions_file:
        for dim in harmony_message.subset.dimensions:
            if dim.name in required_dimensions:
                dim_is_valid = True
            elif dim.name[0] != '/' and f'/{dim.name}' in required_dimensions:
                dim.name = f'/{dim.name}'
                dim_is_valid = True
            else:
                dim_is_valid = False

            if dim_is_valid:
                # Try to extract bounds metadata:
                bounds_array = get_dimension_bounds(dim.name, varinfo, dimensions_file)
                # Retrieve index ranges for the specifically named dimension:
                dim_index_ranges[dim.name] = get_dimension_index_range(
                    dimensions_file[dim.name][:],
                    dim.min,
                    dim.max,
                    bounds_values=bounds_array,
                )
            else:
                # This requested dimension is not in the required dimension set
                raise InvalidNamedDimension(dim.name)

    return dim_index_ranges


def get_dimension_bounds(
    dimension_name: str, varinfo: VarInfoFromDmr, prefetch_dataset: Dataset
) -> MaskedArray:
    """Check if a named dimension has a `bounds` metadata attribute, if so
    retrieve the array of values for the named variable from the NetCDF-4
    variables retrieved from OPeNDAP in the prefetch request.

    If there is no `bounds` reference, or if the variable contained in the
    `bounds` reference is not present in the prefetch output, `None` will
    be returned.

    """
    try:
        # For pseudo-variables, `varinfo.get_variable` returns `None` and
        # therefore has no `references` attribute.
        bounds = varinfo.get_variable(dimension_name).references.get('bounds')
    except AttributeError:
        bounds = None

    if bounds is not None:
        try:
            bounds_data = prefetch_dataset[list(bounds)[0]][:]
        except IndexError:
            # The referred to variable in `bounds` can't be found in dataset
            bounds_data = None
    else:
        bounds_data = None

    return bounds_data


def is_almost_in(value: float, array: np.ndarray) -> bool:
    """Check if a specific value is within the supplied array. The comparison
    will first derive a precision from the smallest difference in elements
    in the supplied array. The comparison will use the minimum value of
    either 10**-5 or (10**-3 * minimum_difference).

    `np.isclose` calculates tolerance = (atol + (rtol * abs(b)), where
    b is the element in the second array being compared. To ensure large
    values don't lose precision, rtol is set to zero below.

    This function was specifically written to help support the ECCO Ocean
    Velocity collection, which has a depth dimension, Z. Most of these
    dimension values are likely set at depths that correspond to specific
    pressure values. The relationship between these (P = rho.g.h) means
    that well rounded pressure values lead to depths without nicely rounded
    values.

    """
    array_precision = min(np.nanmin(np.abs(np.diff(array) / 1000.0)), 0.00001)
    return np.any(
        np.isclose(array, np.full_like(array, value), rtol=0, atol=array_precision)
    )
