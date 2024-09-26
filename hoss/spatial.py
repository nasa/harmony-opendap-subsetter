""" This module includes functions that support bounding box spatial subsets.
    This includes both geographically gridded data and projected grids.

    Using prefetched dimension variables, in full, from the OPeNDAP
    server, and the index ranges that correspond to the regions of these
    variables within the specified bounding box are identified. These index
    ranges are then returned to the `hoss.subset.subset_granule` function to
    be combined with any other index ranges (e.g., temporal).

    If the bounding box crosses the longitudinal edge of the grid for a
    geographically gridded granule, the full longitudinal range of each
    variable is retrieved. The ranges of data for each variable outside of the
    bounding box are set to the variable fill value.

    An example of this would be for the RSSMIF16D data which have a
    grid with 0 ≤ longitude (degrees) < 360. The Harmony message will specify
    a bounding box within -180 ≤ longitude (degrees) < 180. If the western edge
    is west of the Prime Meridian and the eastern edge is east of it, then the
    box will cross the RSSMIF16D grid edge.

    For example: [W, S, E, N] = [-20, -90, 20, 90]

"""

from typing import List, Set

from harmony.message import Message
from netCDF4 import Dataset
from numpy.ma.core import MaskedArray
from varinfo import VarInfoFromDmr

from hoss.bbox_utilities import (
    BBox,
    get_geographic_bbox,
    get_harmony_message_bbox,
    get_shape_file_geojson,
)
from hoss.dimension_utilities import (
    IndexRange,
    IndexRanges,
    get_coordinate_variables,
    get_dimension_bounds,
    get_dimension_extents,
    get_dimension_index_range,
    update_dimension_variables,
)
from hoss.projection_utilities import (
    get_projected_x_y_extents,
    get_projected_x_y_variables,
    get_variable_crs,
)


def get_spatial_index_ranges(
    required_variables: Set[str],
    varinfo: VarInfoFromDmr,
    dimensions_path: str,
    harmony_message: Message,
    shape_file_path: str = None,
) -> IndexRanges:
    """Return a dictionary containing indices that correspond to the minimum
    and maximum extents for all horizontal spatial coordinate variables
    that support all end-user requested variables. This includes both
    geographic and projected horizontal coordinates:

    index_ranges = {'/latitude': (12, 34), '/longitude': (56, 78),
                    '/x': (20, 42), '/y': (31, 53)}

    If geographic dimensions are present and only a shape file has been
    specified, a minimally encompassing bounding box will be found in order
    to determine the longitude and latitude extents.

    For projected grids, coordinate dimensions must be considered in x, y
    pairs. The minimum and/or maximum values of geographically defined
    shapes in the target projected grid may be midway along an exterior
    edge of the shape, rather than a known coordinate vertex. For this
    reason, a minimum grid resolution in geographic coordinates will be
    determined for each projected coordinate variable pairs. The input
    bounding box or shape file will be populated with additional points
    around the exterior of the user-defined GeoJSON shape, to ensure the
    correct extents are derived.

    For projected grids which do not follow CF standards, the projected
    dimension scales are computed based on the values in the coordinate
    datasets if they are available. The geocorners are obtained from the
    coordinate datasets and converted to projected meters based on the crs
    of the product. The dimension scales are then computed based on the
    grid size and grid resolution

    """
    bounding_box = get_harmony_message_bbox(harmony_message)
    index_ranges = {}

    geographic_dimensions = varinfo.get_geographic_spatial_dimensions(
        required_variables
    )
    projected_dimensions = varinfo.get_projected_spatial_dimensions(required_variables)
    non_spatial_variables = required_variables.difference(
        varinfo.get_spatial_dimensions(required_variables)
    )

    with Dataset(dimensions_path, 'r') as dimensions_file:
        if geographic_dimensions:
            # If there is no bounding box, but there is a shape file, calculate
            # a bounding box to encapsulate the GeoJSON shape:
            if bounding_box is None and shape_file_path is not None:
                geojson_content = get_shape_file_geojson(shape_file_path)
                bounding_box = get_geographic_bbox(geojson_content)

            for dimension in geographic_dimensions:
                index_ranges[dimension] = get_geographic_index_range(
                    dimension, varinfo, dimensions_file, bounding_box
                )
        if projected_dimensions:
            for non_spatial_variable in non_spatial_variables:
                index_ranges.update(
                    get_projected_x_y_index_ranges(
                        non_spatial_variable,
                        varinfo,
                        dimensions_file,
                        index_ranges,
                        bounding_box=bounding_box,
                        shape_file_path=shape_file_path,
                    )
                )
        if (not geographic_dimensions) and (not projected_dimensions):
            coordinate_variables = get_coordinate_variables(varinfo, required_variables)
            if coordinate_variables:
                for non_spatial_variable in non_spatial_variables:
                    index_ranges.update(
                        get_projected_x_y_index_ranges(
                            non_spatial_variable,
                            varinfo,
                            dimensions_file,
                            index_ranges,
                            bounding_box=bounding_box,
                            shape_file_path=shape_file_path,
                            override_dimensions=coordinate_variables,
                        )
                    )
        return index_ranges


def get_projected_x_y_index_ranges(
    non_spatial_variable: str,
    varinfo: VarInfoFromDmr,
    dimensions_file: Dataset,
    index_ranges: IndexRanges,
    bounding_box: BBox = None,
    shape_file_path: str = None,
    override_dimensions: Set[str] = set(),
) -> IndexRanges:
    """This function returns a dictionary containing the minimum and maximum
    index ranges for a pair of projection x and y coordinates, e.g.:

    index_ranges = {'/x': (20, 42), '/y': (31, 53)}

    First, the dimensions of the input, non-spatial variable are checked
    for associated projection x and y coordinates. If these are present,
    and they have not already been added to the `index_ranges` cache, the
    extents of the input spatial subset are determined in these projected
    coordinates. This requires the derivation of a minimum resolution of
    the target grid in geographic coordinates. Points must be placed along
    the exterior of the spatial subset shape. All points are then projected
    from a geographic Coordinate Reference System (CRS) to the target grid
    CRS. The minimum and maximum values are then derived from these
    projected coordinate points.

    """
    if not override_dimensions:
        projected_x, projected_y = get_projected_x_y_variables(
            varinfo, non_spatial_variable
        )
    else:
        projected_x = 'projected_x'
        projected_y = 'projected_y'
        override_dimensions_file = update_dimension_variables(
            dimensions_file,
            override_dimensions,
            varinfo,
        )
        dimensions_file = override_dimensions_file
    if (
        projected_x is not None
        and projected_y is not None
        and not set((projected_x, projected_y)).issubset(set(index_ranges.keys()))
    ):
        crs = get_variable_crs(non_spatial_variable, varinfo)

        x_y_extents = get_projected_x_y_extents(
            dimensions_file[projected_x][:],
            dimensions_file[projected_y][:],
            crs,
            shape_file=shape_file_path,
            bounding_box=bounding_box,
        )

        x_bounds = get_dimension_bounds(projected_x, varinfo, dimensions_file)
        y_bounds = get_dimension_bounds(projected_y, varinfo, dimensions_file)
        x_index_ranges = get_dimension_index_range(
            dimensions_file[projected_x][:],
            x_y_extents['x_min'],
            x_y_extents['x_max'],
            bounds_values=x_bounds,
        )
        y_index_ranges = get_dimension_index_range(
            dimensions_file[projected_y][:],
            x_y_extents['y_min'],
            x_y_extents['y_max'],
            bounds_values=y_bounds,
        )
        x_y_index_ranges = {projected_x: x_index_ranges, projected_y: y_index_ranges}
    else:
        x_y_index_ranges = {}

    return x_y_index_ranges


def get_geographic_index_range(
    dimension: str,
    varinfo: VarInfoFromDmr,
    dimensions_file: Dataset,
    bounding_box: BBox,
) -> IndexRange:
    """Extract the indices that correspond to the minimum and maximum extents
    for a specific geographic dimension (longitude or latitude). For
    longitudes, it is assumed that the western extent should be considered
    the minimum extent. If the bounding box crosses a longitude
    discontinuity this will be later identified by the minimum extent index
    being larger than the maximum extent index.

    The return value from this function is an `IndexRange` tuple of format:
    (minimum_index, maximum_index).

    """
    variable = varinfo.get_variable(dimension)
    bounds = get_dimension_bounds(dimension, varinfo, dimensions_file)

    if variable.is_latitude():
        # dimension_utilities.get_dimension_index_range will determine if the
        # dimension is ascending or descending, and flip the North and South
        # values if the dimension is descending.
        minimum_extent = bounding_box.south
        maximum_extent = bounding_box.north
    else:
        # Convert the bounding box western and eastern extents to match the
        # valid range of the dimension data, either of:
        #
        # * -180 ≤ longitude (degrees east) ≤ 180.
        # * 0 ≤ longitude (degrees east) ≤ 360.
        #
        # dimension_utilities.get_dimension_index_range will determine if the
        # dimension is ascending or descending and flip the East and West
        # values if the dimension is descending.
        minimum_extent, maximum_extent = get_bounding_box_longitudes(
            bounding_box, dimensions_file[dimension][:]
        )

    return get_dimension_index_range(
        dimensions_file[dimension][:],
        minimum_extent,
        maximum_extent,
        bounds_values=bounds,
    )


def get_bounding_box_longitudes(
    bounding_box: BBox, longitude_array: MaskedArray
) -> List[float]:
    """Ensure the bounding box extents are compatible with the range of the
    longitude variable. The Harmony bounding box values are expressed in
    the range from -180 ≤ longitude (degrees east) ≤ 180, whereas some
    collections have grids with discontinuities at the Prime Meridian and
    others have sub-pixel wrap-around at the Antimeridian.

    """
    min_longitude, max_longitude = get_dimension_extents(longitude_array)

    western_box_extent = get_longitude_in_grid(
        min_longitude, max_longitude, bounding_box.west
    )
    eastern_box_extent = get_longitude_in_grid(
        min_longitude, max_longitude, bounding_box.east
    )

    return [western_box_extent, eastern_box_extent]


def get_longitude_in_grid(grid_min: float, grid_max: float, longitude: float) -> float:
    """Ensure that a longitude value from the bounding box extents is within
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
