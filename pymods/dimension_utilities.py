""" This module contains utility functions for dimension variables, including
    the conversion from user-defined extents to array indices, and filling of
    variables if the requested range crosses a grid edge.

    These utilities are compatible with any type of dimension. If specific
    handling is required for the extents or dimension values themselves, that
    handling should occur prior to calling functions from this module. For
    example, ensuring requested bounding box longitudes are wrapped or
    unwrapped in accordance with the longitude dimension values.

"""
from datetime import datetime
from logging import Logger
from typing import Dict, List, Optional, Set, Tuple

from netCDF4 import Dataset
from numpy.ma.core import MaskedArray
import numpy as np

from harmony.message import Dimension
from harmony.util import Config
from varinfo import VarInfoFromDmr

from pymods.bbox_utilities import BBox
from pymods.exceptions import InvalidNamedDimension
from pymods.utilities import get_opendap_nc4, format_variable_set_string


IndexRange = Tuple[int]
IndexRanges = Dict[str, IndexRange]


def is_index_subset(bounding_box: Optional[BBox],
                    shape_file_path: Optional[str],
                    dim_request: Optional[List[Dimension]],
                    temporal_range: Optional[List[datetime]]) -> bool:
    """ Determine if the inbound Harmony request specified any parameters that
        will require an index range subset. These will be:

        * Bounding box spatial requests
        * Shape file spatial requests
        * Temporal requests
        * General dimension range subsetting requests

    """
    return any(subset_parameters is not None
               for subset_parameters in [bounding_box, shape_file_path,
                                         dim_request, temporal_range])


def prefetch_dimension_variables(opendap_url: str, varinfo: VarInfoFromDmr,
                                 required_variables: Set[str], output_dir: str,
                                 logger: Logger, access_token: str,
                                 config: Config) -> str:
    """ Determine the dimensions that need to be "pre-fetched" from OPeNDAP in
        order to derive index ranges upon them. Initially, this was just
        spatial and temporal dimensions, but to support generic dimension
        subsets, all required dimensions must be prefetched.

    """
    required_dimensions = varinfo.get_required_dimensions(required_variables)

    logger.info('Variables being retrieved in prefetch request: '
                f'{format_variable_set_string(required_dimensions)}')
    return get_opendap_nc4(opendap_url, required_dimensions, output_dir,
                           logger, access_token, config)


def is_dimension_ascending(dimension: MaskedArray) -> bool:
    """ Read the array associated with a dimension variable and check if the
        variables ascend starting from the zeroth element or not.

    """
    first_index, last_index = np.ma.flatnotmasked_edges(dimension)
    return dimension[first_index] < dimension[last_index]


def get_dimension_index_range(dimension_values: MaskedArray,
                              requested_min: float,
                              requested_max: float) -> IndexRange:
    """ Ensure that both a minimum and maximum value are defined from the
        message, if not, use the first or last value in the dimension array,
        accordingly. The minimum and maximum values must be ordered to be
        ascending or descending in a way that matches the dimension index
        values.

        Once the minimum and maximum values are determined, and sorted in the
        same order as the dimension array values, retrieve the index values
        that correspond to the requested dimension values.

    """
    if is_dimension_ascending(dimension_values):
        dimension_min = requested_min or dimension_values[0]
        dimension_max = requested_max or dimension_values[-1]
    else:
        dimension_min = requested_max or dimension_values[0]
        dimension_max = requested_min or dimension_values[-1]

    return get_dimension_min_max_indices(dimension_values, dimension_min,
                                         dimension_max)


def get_dimension_min_max_indices(dimension: MaskedArray,
                                  minimum_extent: float,
                                  maximum_extent: float) -> IndexRange:
    """ Find the indices closest to the interpolated values of the minimum and
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

    raw_indices = np.interp(dimension_range, dimension_values,
                            dimension_indices)

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


def add_index_range(variable_name: str, varinfo: VarInfoFromDmr,
                    index_ranges: IndexRanges) -> str:
    """ Append the index ranges of each dimension for the specified variable.
        If there are no dimensions with listed index ranges, then the full
        variable should be requested, and no index notation is required.
        A variable with a bounding box crossing the edge of the grid (e.g., at
        the Antimeridian or Prime Meridian) will have a minimum index greater
        than the maximum index. In this case the full dimension range should be
        requested, as the related values will be masked before returning the
        output to the user.

    """
    variable = varinfo.get_variable(variable_name)

    range_strings = []

    for dimension in variable.dimensions:
        dimension_range = index_ranges.get(dimension)

        if dimension_range is not None and dimension_range[0] <= dimension_range[1]:
            range_strings.append(f'[{dimension_range[0]}:{dimension_range[1]}]')
        else:
            range_strings.append('[]')

    if all(range_string == '[]' for range_string in range_strings):
        indices_string = ''
    else:
        indices_string = ''.join(range_strings)

    return f'{variable_name}{indices_string}'


def get_fill_slice(dimension: str, fill_ranges: IndexRanges) -> slice:
    """ Check the dictionary of dimensions that need to be filled for the
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
        fill_slice = slice(fill_ranges[dimension][1] + 1,
                           fill_ranges[dimension][0])
    else:
        fill_slice = slice(None)

    return fill_slice


def get_dimension_extents(dimension_array: np.ndarray) -> Tuple[float]:
    """ Fit the dimension with a straight line, and find the outer edge of the
        first and last pixel, assuming the supplied values lie at the centre of
        each pixel.

    """
    dimension_indices = np.arange(dimension_array.size)
    gradient, _ = np.polyfit(dimension_indices, dimension_array, 1)

    min_extent = dimension_array.min() - np.abs(gradient) / 2.0
    max_extent = dimension_array.max() + np.abs(gradient) / 2.0

    return (min_extent, max_extent)


def get_requested_index_ranges(required_variables: Set[str],
                               varinfo: VarInfoFromDmr,
                               dimensions_path: str,
                               dim_request: List[Dimension]) -> IndexRanges:
    """ Examines the requested dimension names and ranges and extracts the
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
        for dim in dim_request:
            if dim.name in required_dimensions:
                dim_is_valid = True
            elif dim.name[0] != '/' and f'/{dim.name}' in required_dimensions:
                dim.name = f'/{dim.name}'
                dim_is_valid = True
            else:
                dim_is_valid = False

            if dim_is_valid:
                # Retrieve index ranges for the specifically named dimension:
                dim_index_ranges[dim.name] = get_dimension_index_range(
                    dimensions_file[dim.name][:], dim.min, dim.max
                )
            else:
                # This requested dimension is not in the required dimension set
                raise InvalidNamedDimension(dim.name)

    return dim_index_ranges
