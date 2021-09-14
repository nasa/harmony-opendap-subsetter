""" This module includes functions that support bounding box spatial subsets.
    Currently, this includes only geographically gridded data.

    Using prefetched dimension variables, in full, from the OPeNDAP
    server, and the index ranges that correspond to the regions of these
    variables within the specified bounding box are identified. These index
    ranges are then returned to the `pymods.subset.subset_granule` function to
    be combined with any other index ranges (e.g., temporal).

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
from typing import List, Set

from netCDF4 import Dataset
from numpy.ma.core import MaskedArray
from varinfo import VarInfoFromDmr, VariableFromDmr
import numpy as np


from pymods.dimension_utilities import (get_dimension_index_range, IndexRanges,
                                        is_dimension_ascending)


def get_geographic_index_ranges(required_variables: Set[str],
                                varinfo: VarInfoFromDmr, dimensions_path: str,
                                bounding_box: List[float]) -> IndexRanges:
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
    geographic_dimensions = varinfo.get_spatial_dimensions(required_variables)

    with Dataset(dimensions_path, 'r') as dimensions_file:
        for dimension in geographic_dimensions:
            variable = varinfo.get_variable(dimension)
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
