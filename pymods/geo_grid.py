""" This module includes functions that support bounding box spatial subsets of
    geographically gridded data.

    For a given list of required variables, as determined by the `sds-varinfo`
    package, dimensions are first identified. Those dimensions that are
    geographic are determined by checking their `units` metadata attribute.
    The geographic dimension variables are fetched, in full, from the OPeNDAP
    server, and the index ranges that correspond to the regions of these
    variables within the specified bounding box are identified. All required
    variables are then requested from the OPeNDAP server, only in the range of
    the bounding box.

    If the bounding box crosses the longitudinal edge of the grid, the full
    longitudinal range of each variable is retrieved. The ranges of data for
    each variable outside of the bounding box are set to the variable fill
    value.

    An example of this would be for the RSSMIF16D data which have a
    grid with 0 ≤ longitude (degrees) < 360. The Harmony message will specify
    a bounding box within -180 ≤ longitude (degrees) < 180. If the western edge
    is west of the Prime Meridian and the eastern edge is east of it, then the
    box will cross the RSSMIF16D grid edge.

    For example: [W, S, E, N] = [-20, -90, 20, 90]

"""
from logging import Logger
from typing import Dict, List, Set

from netCDF4 import Dataset
from numpy.ma.core import MaskedArray
import numpy as np

from harmony.util import Config
from varinfo import VarInfoFromDmr, VariableFromDmr

from pymods.utilities import get_opendap_nc4


def get_geo_bounding_box_subset(required_variables: Set[str],
                                dataset: VarInfoFromDmr,
                                bounding_box: List[float], url: str,
                                output_dir: str, logger: Logger,
                                access_token: str, config: Config) -> str:
    """ Inspect the list of required variables to determine the geographic
        dimensions. Then perform a pre-fetch of data to retrieve the full range
        of those variables. From the full range, determine the index ranges for
        the coordinate dimensions.

        Iterate through each variable and determine the index ranges for each.
        If a dimension is a geographic coordinate, then use the index range
        determined from the previous steps, otherwise retrieve the full range.

        If a request includes a bounding box crossing a longitude discontinuity
        at either the Prime Meridian or Antimeridian, the western and eastern
        extents of the bounding box will seem to be in the reverse order. In
        this case, the full range of longitudes is requested for all variables
        and the data outside of the bounding box are filled.

    """
    geographic_dimensions = dataset.get_spatial_dimensions(required_variables)

    if len(geographic_dimensions) > 0:
        # Prefetch geographic dimension data from OPeNDAP
        dimensions_path = get_opendap_nc4(url, geographic_dimensions,
                                          output_dir, logger, access_token,
                                          config)

        # Derive index ranges of the bounding box for each geographic variable
        index_ranges = get_dimension_index_ranges(dimensions_path, dataset,
                                                  geographic_dimensions,
                                                  bounding_box)
    else:
        # There are no geographic dimensions for the required variables, so
        # none of the variables have index ranges constraints.
        index_ranges = {}

    # Cycle through all variables and map to correct dimension index ranges
    variables_with_ranges = set(
        add_index_range(variable, dataset, index_ranges)
        for variable in required_variables
    )

    # Get full variable subset using derived index ranges:
    output_path = get_opendap_nc4(url, variables_with_ranges, output_dir,
                                  logger, access_token, config)

    # Filter the geographic index ranges to only include those to be filled
    fill_ranges = {dimension: index_range
                   for dimension, index_range
                   in index_ranges.items()
                   if index_range[0] > index_range[1]}

    # If variables need filling (due to crossing longitude discontinuity)
    # fill the gaps between the ranges in the bounding box
    if len(fill_ranges) > 0:
        fill_variables(output_path, dataset, required_variables, fill_ranges)

    return output_path


def get_dimension_index_ranges(dimensions_path: str, dataset: VarInfoFromDmr,
                               geographic_dimensions: Set[str],
                               bounding_box: List[float]) -> Dict[str, List[int]]:
    """ Iterate through all geographic dimensions and extract the indices that
        correspond to the minimum and maximum extents in that dimension. For
        longitudes, it is assumed that the western extent should be considered
        the minimum extent. If the bounding box crosses a longitude
        discontinuity this will be later identified by the minimum extent index
        being larger than the maximum extent index.

        The return value from this function is a dictionary that contains the
        index ranges for each geographic dimension, such as:

        index_range = {'/latitude': [12, 34], '/longitude': [56, 78]}

    """
    index_ranges = {}

    with Dataset(dimensions_path, 'r') as dimensions_file:
        for dimension in geographic_dimensions:
            variable = dataset.get_variable(dimension)
            if variable.is_latitude():
                if is_dimension_ascending(dimensions_file[dimension][:]):
                    # dimension array runs -90 to 90 degrees.
                    # The minimum index will be the south extent
                    minimum_extent = bounding_box[1]
                    maximum_extent = bounding_box[3]
                else:
                    # dimension array runs 90 to -90 degrees.
                    # The minimum index will be the north extent
                    minimum_extent = bounding_box[3]
                    maximum_extent = bounding_box[1]
            else:
                # First, convert the bounding box western and eastern extents
                # to match the valid range of the dimension data
                west_extent, east_extent = get_bounding_box_longitudes(
                    bounding_box, dimensions_file[dimension][:], variable
                )
                if is_dimension_ascending(dimensions_file[dimension][:]):
                    # dimension array runs -180 to 180 (or 0 to 360) degrees.
                    # The minimum index will be the west extent
                    minimum_extent = west_extent
                    maximum_extent = east_extent
                else:
                    # dimension array runs 180 to -180 (or 360 to 0) degrees.
                    # The minimum index will be the east extent
                    minimum_extent = east_extent
                    maximum_extent = west_extent

            index_ranges[dimension] = get_dimension_index_range(
                dimensions_file[dimension][:], minimum_extent, maximum_extent
            )

    return index_ranges


def is_dimension_ascending(dimension: MaskedArray) -> bool:
    """ Read the array associated with a dimension variable and check if the
        variables ascend starting from the zeroth element or not.

    """
    first_index, last_index = np.ma.flatnotmasked_edges(dimension)
    return dimension[first_index] < dimension[last_index]


def get_dimension_index_range(dimension: MaskedArray, minimum_extent: float,
                              maximum_extent: float) -> List[int]:
    """ Find the indices closest to the interpolated values of the minimum and
        maximum extents in that dimension.

        Use of `numpy.interp` maps the dimension scale values to their index
        values and then computes an interpolated index value that best matches
        the bounding value (minimum_extent, maximum_extent) to a "fractional"
        index value. Rounding that value gives the starting index value for the
        cell that contains that bound.

        For a latitude dimension:

        * `minimum_extent` is the southern extent of the bounding box.
        * `maximum_extent` is the northern extent of the bounding box.

        For a longitude dimension:

        * `minimum_extent` is the western extent of the bounding box.
        * `maximum_extent` is the eastern extent of the bounding box.

        The input longitude extent values must conform to the valid range of
        the native dimension data.

    """
    dimension_range = [minimum_extent, maximum_extent]
    dimension_indices = np.arange(dimension.size)

    if is_dimension_ascending(dimension):
        dimension_values = dimension
    else:
        # second argument to `np.linterp` must be ascending.
        # The dimension indices also should be flipped to still be correct
        dimension_values = np.flip(dimension)
        dimension_indices = np.flip(dimension_indices)

    raw_indices = np.interp(dimension_range, dimension_values,
                            dimension_indices)

    if raw_indices[0] % 1 == 0.5:
        # Minimum extent is exactly halfway between two pixels, round up.
        raw_minimum_index = np.nextafter(raw_indices[0], raw_indices[0] + 1)
    else:
        raw_minimum_index = raw_indices[0]

    if raw_indices[1] % 1 == 0.5:
        # Maximum extent is exactly halfway between two pixels, round down.
        raw_maximum_index = np.nextafter(raw_indices[1], raw_indices[1] - 1)
    else:
        raw_maximum_index = raw_indices[1]

    minimum_index = int(np.rint(raw_minimum_index))
    maximum_index = int(np.rint(raw_maximum_index))

    return [minimum_index, maximum_index]


def get_bounding_box_longitudes(bounding_box: List[float],
                                longitude_array: MaskedArray,
                                longitude: VariableFromDmr) -> List[float]:
    """ Ensure the bounding box longitude extents are in the valid range for
        the longitude variable. The bounding box values are expected to range
        from -180 ≤ longitude (degrees) < 180, whereas some collections have
        grids with discontinuities at the Prime Meridian.

        The bounding box from the Harmony message is ordered: [W, S, E, N]

    """
    valid_range = get_valid_longitude_range(longitude, longitude_array)

    if valid_range[1] > 180:
        # Discontinuity at Prime Meridian: 0 ≤ longitude (degrees) < 360
        western_box_extent = unwrap_longitude(bounding_box[0])
        eastern_box_extent = unwrap_longitude(bounding_box[2])
    else:
        # Discontinuity at Antimeridian: -180 ≤ longitude (degrees) < 180
        western_box_extent = bounding_box[0]
        eastern_box_extent = bounding_box[2]

    return [western_box_extent, eastern_box_extent]


def wrap_longitude(longitude: float) -> float:
    """ Wrap longitude to be in the -180 ≤ longitude (degrees) < 180 range.
        For longitudes already in this range, this is a no-op.

    """
    return ((longitude + 180) % 360) - 180


def unwrap_longitude(wrapped_longitude: float) -> float:
    """ Convert longitude from the -180 ≤ longitude (degrees) < 180 range to
        0 ≤ longitude (degrees) < 360. This allows that bounding box to be
        converted from its native range to match that of collections in this
        latter format (e.g., RSSMIF16D). The bounding box needs to be evaluated
        in the same range as the collection's grid, to ensure the longitude
        discontinuity is preserved and discontinuous array indices can be
        identified.

    """
    return ((wrapped_longitude % 360) + 360) % 360


def get_valid_longitude_range(longitude: VariableFromDmr,
                              longitude_array: MaskedArray) -> List[float]:
    """ Check the variable metadata for `valid_range` or `valid_min` and
        `valid_max`. If no metadata data attributes indicating the valid range
        are present, check if the data contain a value in the range
        180 < longitude < 360 to determine the adopted convention.

        The expected options are:

        * Discontinuity at Antimeridian: -180 ≤ longitude (degrees) < 180
        * Discontinuity at Prime Meridian: 0 ≤ longitude (degrees) < 360

    """
    valid_range = longitude.get_range()

    if valid_range is None and np.max(longitude_array) > 180.0:
        valid_range = [0.0, 360.0]
    elif valid_range is None:
        valid_range = [-180.0, 180.0]

    return valid_range


def add_index_range(variable_name: str, dataset: VarInfoFromDmr,
                    index_ranges: Dict[str, List[int]]) -> str:
    """ Append the index ranges of each dimension for the specified variable.
        If there are no dimensions with listed index ranges, then the full
        variable should be requested, and no index notation is required.
        A variable with a bounding box crossing the edge of the grid (e.g., at
        the Antimeridian or Prime Meridian) will have a minimum index greater
        than the maximum index. In this case the full dimension range should be
        requested, as the related values will be masked before returning the
        output to the user.

    """
    variable = dataset.get_variable(variable_name)

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


def fill_variables(output_path: str, dataset: VarInfoFromDmr,
                   required_variables: Set[str],
                   fill_ranges: Dict[str, List[int]]) -> None:
    """ Cycle through the output NetCDF-4 file and check the dimensions of
        each variable. If the minimum index is greater than the maximum index
        in the subset range, then the requested bounding box crossed an edge of
        the grid, and must be filled in between those values.

        This function will only be called if there are variables that require
        filling.

        Note - longitude variables themselves will not be filled, to ensure
        valid grid coordinates at all points of the science variables.

    """
    dimensions_to_fill = set(fill_ranges)

    with Dataset(output_path, 'a', format='NETCDF4') as output_dataset:
        for variable_path in required_variables:
            variable = dataset.get_variable(variable_path)
            if (
                    not variable.is_longitude() and
                    len(dimensions_to_fill.intersection(variable.dimensions)) > 0
            ):
                fill_index_tuple = tuple(
                    get_fill_slice(dimension, fill_ranges)
                    for dimension in variable.dimensions
                )

                output_dataset[variable_path][fill_index_tuple] = np.ma.masked


def get_fill_slice(dimension: str, fill_ranges: Dict[str, List[int]]) -> slice:
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
