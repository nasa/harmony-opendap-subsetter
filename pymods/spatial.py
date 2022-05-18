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
from varinfo import VarInfoFromDmr


from pymods.bbox_utilities import BBox
from pymods.dimension_utilities import (get_dimension_extents,
                                        get_dimension_index_range, IndexRanges,
                                        is_dimension_ascending)


def get_geographic_index_ranges(required_variables: Set[str],
                                varinfo: VarInfoFromDmr, dimensions_path: str,
                                bounding_box: BBox) -> IndexRanges:
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
                    minimum_extent = bounding_box.south
                    maximum_extent = bounding_box.north
                else:
                    # dimension array runs 90 to -90 degrees.
                    # The minimum index will be the north extent
                    minimum_extent = bounding_box.north
                    maximum_extent = bounding_box.south
            else:
                # First, convert the bounding box western and eastern extents
                # to match the valid range of the dimension data
                west_extent, east_extent = get_bounding_box_longitudes(
                    bounding_box, dimensions_file[dimension][:]
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


def get_bounding_box_longitudes(bounding_box: BBox,
                                longitude_array: MaskedArray) -> List[float]:
    """ Ensure the bounding box extents are compatible with the range of the
        longitude variable. The Harmony bounding box values are expressed in
        the range from -180 ≤ longitude (degrees east) ≤ 180, whereas some
        collections have grids with discontinuities at the Prime Meridian and
        others have sub-pixel wrap-around at the Antimeridian.

    """
    min_longitude, max_longitude = get_dimension_extents(longitude_array)

    western_box_extent = get_longitude_in_grid(min_longitude, max_longitude,
                                               bounding_box.west)
    eastern_box_extent = get_longitude_in_grid(min_longitude, max_longitude,
                                               bounding_box.east)

    return [western_box_extent, eastern_box_extent]


def get_longitude_in_grid(grid_min: float, grid_max: float,
                          longitude: float) -> float:
    """ Ensure that a longitude value from the bounding box extents is within
        the full longitude range of the grid. If it is not, check the same
        value +/- 360 degrees, to see if either of those are present in the
        grid. This function returns the value of the three options that lies
        within the grid. If none of these values are within the grid, then the
        original longitude value is returned.

        This functionality is used for grids where the longitude values are not
        -180 ≤ longitude (degrees east) ≤ 180. This includes:

        * RSSMIF16D: 0 ≤ longitude (degrees east) ≤ 360.
        * MERRA-2 products:  -180.3125 ≤ longitude (degrees east) ≤ 179.6875.

    """
    decremented_longitude = longitude - 360
    incremented_longitude = longitude + 360

    if grid_min <= longitude <= grid_max:
        adjusted_longitude = longitude
    elif grid_min <= decremented_longitude <= grid_max:
        adjusted_longitude = decremented_longitude
    elif grid_min <= incremented_longitude <= grid_max:
        adjusted_longitude = incremented_longitude
    else:
        # None of the values are in the grid, so return the original value.
        adjusted_longitude = longitude

    return adjusted_longitude
