"""This module contains utility functionality relating to projected grids.
Primarily, these functions will determine the minimum horizontal spatial
resolution in geographic coordinates of an input grid from the x and y
dimension variables. With this resolution in hand, further functionality
will ensure that any specified GeoJSON shape is populated with points at
that resolution. These points can then be projected to the target grid in
order to determine the full extent of the GeoJSON input in the target grid.
This extent will then determine the index ranges requested from OPeNDAP for
projected grids.

"""

import json
from typing import Dict, List, Optional, Tuple, Union, get_args

import numpy as np
import dask.array as da
from dask.array.overlap import overlap as da_overlap

from pyproj import CRS, Transformer
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    shape,
)
from shapely.geometry.base import BaseGeometry
from varinfo import VarInfoFromDmr

from hoss.bbox_utilities import BBox, flatten_list, BBox_R
from hoss.exceptions import (
    InvalidGranuleDimensions,
    InvalidInputGeoJSON,
    InvalidRequestedRange,
    MissingGridMappingMetadata,
    MissingGridMappingVariable,
    MissingSpatialSubsetInformation,
)

Coordinates = Tuple[float, float]
MultiShape = Union[GeometryCollection, MultiLineString, MultiPoint, MultiPolygon]
Shape = Union[LineString, Point, Polygon, MultiShape]


def get_variable_crs(variable: str, varinfo: VarInfoFromDmr) -> CRS:
    """Retrieves the grid mapping variable metadata attributes for a given
    variable and creates a `pyproj.CRS` object from the grid mapping attributes.
    """
    return CRS.from_cf(get_grid_mapping_attributes(variable, varinfo))


def get_grid_mapping_attributes(variable: str, varinfo: VarInfoFromDmr) -> Dict:
    """Check the metadata attributes for the variable to find the associated
    grid mapping variable.

    All metadata attributes that contain references from one variable to
    another are stored in the `Variable.references` dictionary attribute
    as sets. There should only be one reference in the `grid_mapping`
    attribute value, so the first element of the set is retrieved.
    If the grid mapping variable, as referred to in the grid_mapping
    CF-Convention metadata attribute, does not exist in the file then
    the earthdata-varinfo configuration file is checked, as it may
    contain metadata overrides specified for that non-existent variable
    name.
    """
    var = varinfo.get_variable(variable)
    assert (
        var is not None  # avoids type checking issues
    ), "Program error - variable should never be undefined in VarInfo"

    grid_mapping = next(iter(var.references.get("grid_mapping", [])), None)

    if grid_mapping is not None:
        try:
            grid_mapping_variable = varinfo.get_variable(grid_mapping)
            if grid_mapping_variable is not None:
                cf_attributes = grid_mapping_variable.attributes
            else:
                # check for configuration provided attributes
                cf_attributes = varinfo.get_missing_variable_attributes(grid_mapping)

            if cf_attributes:
                return cf_attributes
            raise MissingGridMappingVariable(grid_mapping, variable)

        except AttributeError as exception:
            raise MissingGridMappingVariable(grid_mapping, variable) from exception

    else:
        raise MissingGridMappingMetadata(variable)


def get_master_geotransform(
    variable: str, varinfo: VarInfoFromDmr
) -> Optional[List[int]]:
    """Retrieves the `master_geotransform` attribute from the grid mapping
    attributes of the given variable. If the `master_geotransform` attribute
    doesn't exist, `None` will be returned.
    """
    return get_grid_mapping_attributes(variable, varinfo).get(
        "master_geotransform", None
    )


def get_config_geo_spatial_extent(
    variable: str, varinfo: VarInfoFromDmr
) -> BBox | None:
    """Retrieves the `geographic_spatial_extent` attribute from the grid mapping
    attributes of the given variable. If the `geographic_spatial_extent` attribute
    doesn't exist, `None` will be returned.
    """
    spatial_extent = get_grid_mapping_attributes(variable, varinfo).get(
        "geographic_spatial_extent", None
    )
    if spatial_extent is not None:
        return BBox(
            spatial_extent[0],  # west
            spatial_extent[1],  # south
            spatial_extent[2],  # east
            spatial_extent[3],  # north
        )
    return None


def get_x_y_dim_var_names(
    varinfo: VarInfoFromDmr, variable: str
) -> Tuple[Optional[str], Optional[str]]:
    """Retrieve the names of the projected x and y dimension variables
    associated with a variable. If either are not found, a `None` value
    will be returned for the absent dimension variable.

    Note - the input variables to this function are only filtered to remove
    variables that are spatial dimensions. The input to this function may
    have no dimensions, or may not be spatially gridded.
    """
    var = varinfo.get_variable(variable)
    assert (
        var is not None  # avoids type checking issues
    ), "Program error - variable should never be undefined in VarInfo"

    x_dim_var = next(
        (
            dimension
            for dimension in var.dimensions
            if is_projection_x_dimension(varinfo, dimension)
        ),
        None,
    )
    y_dim_var = next(
        (
            dimension
            for dimension in var.dimensions
            if is_projection_y_dimension(varinfo, dimension)
        ),
        None,
    )
    return x_dim_var, y_dim_var


def is_projection_x_dimension(varinfo: VarInfoFromDmr, dimension_variable: str) -> bool:
    """Check if the named variable exists in the `VarInfoFromDmr`
    representation of the granule. If so, check the `standard_name`
    attribute conforms to the CF-Convention defined options for a
    projection x coordinate.

    The variable must be first checked to see if it exists as some
    dimensions, such as the `nv`, `latv` and `lonv` that define the
    2-element dimension of bounds variables, exist only as a size, not as a
    full variable within the input granule.

    """
    projected_x_names = ("projection_x_coordinate", "projection_x_angular_coordinate")
    var = varinfo.get_variable(dimension_variable)
    return var is not None and (
        var.attributes.get("standard_name") in projected_x_names
    )


def is_projection_y_dimension(varinfo: VarInfoFromDmr, dimension_variable: str) -> bool:
    """Check if the named variable exists in the representation of the
    granule. If so, check the `standard_name` attribute conforms to the
    CF-Convention defined options for a projection y coordinate.

    The variable must be first checked to see if it exists as some
    dimensions, such as the `nv`, `latv` and `lonv` that define the
    2-element dimension of bounds variables, exist only as a size, not as a
    full variable within the input granule.

    """
    projected_y_names = ("projection_y_coordinate", "projection_y_angular_coordinate")
    var = varinfo.get_variable(dimension_variable)
    return var is not None and (
        var.attributes.get("standard_name") in projected_y_names
    )


def get_projected_x_y_extents(
    x_values: np.ndarray,  # 1D dimension coordinate (scale)
    y_values: np.ndarray,  # 1D dimension coordinate (scale)
    crs: CRS,
    shape_file: str | None = None,
    bounding_box: BBox | None = None,
    config_geo_bbox: BBox | None = None,
) -> Dict[str, float]:  # {'x_min', 'x_max', 'y_min', 'y_max'}
    """Retrieve the minimum and maximum values for a projected grid as derived
    from either a bounding box or GeoJSON shape file, both of which are
    defined in geographic coordinates.

    A minimum grid resolution will be determined in the geographic
    Coordinate Reference System (CRS). The input spatial constraint will
    then have points populated around its exterior at this resolution.
    These geographic points will then all be projected to the target grid
    CRS, allowing the retrieval of a minimum and maximum value in both the
    projected x and projected y dimension.

    Example output: x_y_extents = {'x_min': 1000,
                                   'x_max': 4000,
                                   'y_min': 2500,
                                   'y_max': 5500}
    When projected, the perimeter of a bounding box or polygon given in
    geographic terms may become curved. To determine the X, Y extent of the
    requested bounding area, we need a perimeter with a suitable density of
    points, such that we catch the extent of that curve when projected. The
    calculated resolution is used to define the necessary number of points
    (density).
    """

    geo_bbox, geographic_resolution = get_grid_geographic_info(x_values, y_values, crs)

    # Note - we have to get full reverse-projected lat/lons (in
    # get_grid_geographic_info, call above) to properly determine resolution,
    # but... Now check for configured lat-lon extent to supercede grid-defined
    # lat-lon extent
    if config_geo_bbox is None:
        granule_extent = geo_bbox
    else:
        granule_extent = config_geo_bbox

    print(f"Granule Geo bbox: {granule_extent}, resolution: {geographic_resolution}")

    # Create the "densified" perimeter using either a bounding-box
    # (priority) or a shape_file. One or the other is required.
    densified_perimeter = get_densified_perimeter(
        geographic_resolution, shape_file=shape_file, bounding_box=bounding_box
    )

    # To avoid out-of-limits projection in the last step, we need to
    # clip the bounding perimeter to the source file's geographic extents
    clipped_perimeter = get_filtered_points(densified_perimeter, granule_extent)

    # Getting subset x-y extents is also limited by the source granule's extents
    # in projected space.
    source_granule_extent = {
        "x_min": np.min(x_values),
        "x_max": np.max(x_values),
        "y_min": np.min(y_values),
        "y_max": np.max(y_values),
    }
    return get_x_y_extents_from_geographic_perimeter(
        clipped_perimeter, crs, source_granule_extent
    )


def get_grid_geographic_info(
    x_values: np.ndarray, y_values: np.ndarray, crs: CRS
) -> Tuple[BBox, float]:  # geo-BBox, WSEN, geo-resolution
    """Get the geographic bounding extents (BBox) of the source grid
    and the geographic resolution. Both of these require the projected
    grid converted to geographic coordinates. This method uses Dask.Array
    features to expedite handling large arrays, and calls reduce_geo_chunks
    to process per DASK Array chunk (i.e., get_grid_geographic_info per chunk).
    combine_bbox_array is used to combine the results of "reducing" the chunks.

    Note that "projection" for x & y refers to the extension of values in the x
    & y direction, as well as the data being coordinates in the CRS projection
    space.
    """
    # DASK'ify the x and y coordinate vectors (dask.array as da)
    xda = da.from_array(x_values.data, chunks="1000")
    yda = da.from_array(y_values.data, chunks="1000")

    # Create a 2D array from x values repeated down the y dimension
    projected_x = da.repeat(xda.reshape(1, len(xda)), len(yda), axis=0)

    # Create a 2D array from y values repeated acrss the x dimension
    projected_y = da.repeat(yda.reshape(len(yda), 1), len(xda), axis=1)

    # Combine the projected_x and projected_y arrays into a single array
    # to simplify DASK Array processing (single array vs parallel handling
    # of two arrays)
    projected_x_y = da.stack([projected_x, projected_y], axis=0).rechunk(
        chunks=(2, 1000, 1000)
    )

    # Using the dask.array.overlap feature, extend the chunks with single
    # row/column overlaps at the edges. This allows resolution calculations to
    # include the edges with adjacent values to work with. No extension occurs
    # at the edges.
    projected_x_y = da_overlap(projected_x_y, depth={1: 1, 2: 1}, boundary="none")
    # depth={1:1,2:1} means 1 element overlap in dimensions 1 & 2, row, column.
    # There is no overlap in first dimension (0, the x, y stacking)

    bbox_r = da.reduction(
        projected_x_y,
        chunk=lambda a_chunk, **kwargs: reduce_geo_chunk(a_chunk, crs=crs, **kwargs),
        aggregate=combine_bbox_array,
        axis=(1, 2),  # axes of reduction is (x, y)
        dtype=projected_x.dtype,
        concatenate=False,
    ).compute()

    return (bbox_r.bbox, bbox_r.resolution)


def combine_bbox_r(a: BBox_R, b: BBox_R) -> BBox_R:
    """Reduce a pair of BBox_R to a single BBox_R, returning an encompassing
    BBox and the lessor of the given resolutions
    """
    bboxr = BBox_R(
        BBox(
            min(a.bbox[0], b.bbox[0]),
            min(a.bbox[1], b.bbox[1]),
            max(a.bbox[2], b.bbox[2]),
            max(a.bbox[3], b.bbox[3]),
        ),
        min(a.resolution, b.resolution),
    )
    return bboxr


def combine_bbox_array(bbox_r_arr: BBox_R | List[BBox_R], **kwargs) -> BBox_R:
    """Combine an array of BBox_R results to a single BBox_R
    """
    agg = BBox_R(BBox(180, 90, -90, -180), 180)  # all beyond expected values

    # Return (combine_bbox_r(agg, bbox_r) for bbox_r in bbox_r_arr)
    # but list comprehension creates a "generator" in python, which DASK
    # cannot serialize. Also - it turns out DASK may submit nested arrays
    # of arrays
    if isinstance(bbox_r_arr, BBox_R):
        agg = bbox_r_arr
    elif isinstance(bbox_r_arr, list):
        for bbox_r in bbox_r_arr:
            agg = combine_bbox_r(agg, combine_bbox_array(bbox_r, **kwargs))
    return agg


def reduce_geo_chunk(
    projected_x_y: np.ndarray,
    # stacked array of [ repeated rows of x values (2D),
    #                    repeated columns of y values (2D) ]
    crs: CRS | None = None,
    **kwargs,
) -> BBox_R:
    """Process a projected_x_y array chunk to determine the geographic extents
    (in lat/lon values) and geographic resolution (in degrees) of the x, y
    locations.
    """
    # Unstack arrays
    projected_x = projected_x_y[0]
    projected_y = projected_x_y[1]

    grid_lats, grid_lons = get_grid_lat_lons(projected_x, projected_y, crs)
    if not np.all(np.isfinite(grid_lats)) or not np.all(np.isfinite(grid_lons)):
        raise InvalidGranuleDimensions

    bbox = BBox(grid_lons.min(), grid_lats.min(), grid_lons.max(), grid_lats.max())
    resolution = get_geographic_resolution(grid_lons, grid_lats)

    return BBox_R(bbox, resolution)


def get_grid_lat_lons(
    projected_x: np.ndarray, projected_y: np.ndarray, crs: CRS | None
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform projected_x + projected_y coordinates into lat/lon
    coordinates
    """
    if not crs:
        (lats, lons) = zip(projected_x, projected_y)
        return (np.array(lats), np.array(lons))

    to_geo_transformer = Transformer.from_crs(crs, 4326)

    lats, lons = to_geo_transformer.transform(  # pylint: disable=unpacking-non-sequence
        projected_x, projected_y
    )
    return (lats, lons)


def get_geographic_resolution(longitudes: np.ndarray, latitudes: np.ndarray) -> float:
    """Calculate the distance between diagonally adjacent cells in both
    longitude and latitude. Combine those differences in quadrature to
    obtain Euclidean distances. Return the minimum of these Euclidean
    distances. Over the typical distances being considered, differences
    between the Euclidean and geodesic distance between points should be
    minimal, with Euclidean distances being slightly shorter.

    Note - strictly speaking this is not the "true" resolution, but a
    diagonal cell-wise distance for "densifying" the perimeter points. It
    will be approximately √2 * the cell-center distance, which more typically
    defines the resolution.
    """
    lon_square_diffs = np.square(np.subtract(longitudes[1:, 1:], longitudes[:-1, :-1]))
    lat_square_diffs = np.square(np.subtract(latitudes[1:, 1:], latitudes[:-1, :-1]))
    return np.nanmin(np.sqrt(np.add(lon_square_diffs, lat_square_diffs)))


def get_densified_perimeter(
    resolution: float, shape_file: str | None = None, bounding_box: BBox | None = None
) -> List[Coordinates]:
    """Take a shape file or bounding box, as defined by the input Harmony
    request and return a set of perimeter points - filled to the density
    specified by the resolution given. One of either the bounding_box
    argument or the shape_file argument is required. Priority is given
    to the bounding_box argument. A bounding box is converted to a set
    of perimeter points. A shape_file is resolved the exterior of the
    GeoJson features.
    """
    if bounding_box is not None:
        resolved_geojson = get_resolved_feature(
            get_bbox_polygon(bounding_box), resolution
        )
    elif shape_file is not None:
        with open(shape_file, "r", encoding="utf-8") as file_handler:
            geojson_content = json.load(file_handler)

        resolved_geojson = get_resolved_features(geojson_content, resolution)
    else:
        raise MissingSpatialSubsetInformation()

    return resolved_geojson


def get_filtered_points(
    spatial_constraint_pts: List[Coordinates], granule_extent: BBox
) -> List[Coordinates]:
    """Returns spatial constraint lat/lon values clipped to the spatial
    extent of the granule or raises exception if entirely outside
    the granule
    """
    requested_lons, requested_lats = zip(*spatial_constraint_pts)
    # if all the points in the bounding box are outside the granule extent,
    if (
        np.max(requested_lons) < granule_extent.west
        or np.min(requested_lons) > granule_extent.east
        or np.max(requested_lats) < granule_extent.south
        or np.min(requested_lats) > granule_extent.north
    ):
        raise InvalidRequestedRange

    # If the spatial constraint encloses the granule extent,
    # clip to the granule extent.
    # First, all lon values are clipped within the granule lon extent
    clipped_lons = np.clip(requested_lons, granule_extent.west, granule_extent.east)

    # all lat values are clipped to granule lat extent
    clipped_lats = np.clip(requested_lats, granule_extent.south, granule_extent.north)

    return list(zip(clipped_lons, clipped_lats))


def get_bbox_polygon(bounding_box: BBox) -> Polygon:
    """Convert a bounding box into a polygon with points at each corner of
    that box.
    """
    coordinates = [
        (bounding_box.west, bounding_box.south),
        (bounding_box.east, bounding_box.south),
        (bounding_box.east, bounding_box.north),
        (bounding_box.west, bounding_box.north),
        (bounding_box.west, bounding_box.south),
    ]
    return Polygon(coordinates)


def get_resolved_features(
    geojson_content: Dict, resolution: float
) -> List[Coordinates]:
    """Geojson_content is read and parsed from a json file, yielding
    `shapely.geometry.shape` objects for all features. These features need
    to be resolved (densified and simplified to the exterior perimeter)
    using the supplied resolution of the projected grid.

    * The first condition will recognise a single GeoJSON geometry, using
      the allowed values of the `type` attribute.
    * The second condition will recognise a full GeoJSON feature, which
      will include the `geometry` attribute.
    * The third condition recognises feature collections, and will create a
      `shapely.geometry.shape` object for each child feature.

    Strictly, RFC7946 defines geometry types with capital letters, however,
    this function converts any detected `type` attribute to an entirely
    lowercase string, to avoid missing feature types due to unexpected
    lowercase letters.

    """
    feature_types = (
        "geometrycollection",
        "linestring",
        "point",
        "polygon",
        "multilinestring",
        "multipoint",
        "multipolygon",
    )

    if not geojson_content:  # avoids type check failures for Null
        return []
    if geojson_content.get("type", "").lower() in feature_types:
        resolved_features = get_resolved_feature(shape(geojson_content), resolution)
    elif "geometry" in geojson_content:
        resolved_features = get_resolved_feature(
            shape(geojson_content["geometry"]), resolution
        )
    elif "features" in geojson_content:
        resolved_features = flatten_list(
            [
                get_resolved_feature(shape(feature["geometry"]), resolution)
                for feature in geojson_content["features"]
            ]
        )
    else:
        raise InvalidInputGeoJSON()

    return resolved_features


def get_resolved_feature(
    feature: Shape | BaseGeometry, resolution: float
) -> List[Coordinates]:
    """Take an input `shapely` feature, such as a GeoJSON Point, LineString,
    Polygon or multiple of those options, and return a list of coordinates
    on that feature at the supplied resolution. This resolution corresponds
    to that of a projected grid.

    * For a Polygon, resolve each line segment on the exterior of the
      Polygon. The holes within the polygon should be enclosed by the
      exterior, and therefore should not contain an extreme point in
      spatial extent.
    * For a LineString resolve each line segment and return all points
      along each segment.
    * For a Point object return the input point.
    * For a shape with multiple geometries, recursively call this function
      on each sub-geometry, flattening the multiple lists of points into a
      single list.

    Later processing will try to determine the extents from these points,
    but won't require the list of coordinates to distinguish between input
    subgeometries, so a flattened list of all coordinates is returned.
    """
    if isinstance(feature, Polygon):
        resolved_points = get_resolved_geometry(
            list(feature.exterior.coords), resolution
        )
    elif isinstance(feature, LineString):
        resolved_points = get_resolved_geometry(
            list(feature.coords), resolution, is_closed=feature.is_closed
        )
    elif isinstance(feature, Point):
        resolved_points = [(feature.x, feature.y)]
    elif isinstance(feature, get_args(MultiShape)):
        resolved_points = flatten_list(
            [
                get_resolved_feature(sub_geometry, resolution)
                for sub_geometry in feature.geoms  # type: ignore
                # note: BaseGeometry added as type option above to address
                # type checking, but is not expected.
            ]
        )
    else:
        raise InvalidInputGeoJSON()

    return resolved_points


def get_resolved_geometry(
    geometry_points: List[Tuple[float, ...]],
    # List[Coordinates] & CoordinateSequence (shapely.coords) fail type checking
    resolution: float,
    is_closed: bool = True,
) -> List[Coordinates]:
    """Iterate through all pairs of consecutive points and ensure that, if
    those points are further apart than the resolution of the input data,
    additional points are placed along that edge at regular intervals. Each
    line segment will have regular spacing, and will remain anchored at the
    original start and end of the line segment. This means the spacing of
    the points will have an upper bound of the supplied resolution, but may
    be a shorter distance to account for non-integer multiples of the
    resolution along the line.

    To avoid duplication of points, the last point of each line segment is
    not retained, as this will match the first point of the next line
    segment. For geometries that do not form a closed ring,
    the final point of the geometry is appended to the full list of
    resolved points to ensure all points are represented in the output. For
    closed geometries, this is already present as the first returned point.
    """
    new_points = [
        get_resolved_line(point_one, geometry_points[point_one_index + 1], resolution)[
            :-1
        ]
        for point_one_index, point_one in enumerate(geometry_points[:-1])
    ]
    if not is_closed:
        new_points.append([geometry_points[-1]])

    return flatten_list(new_points)


def get_resolved_line(
    point_one: Coordinates | Tuple[float, ...],
    point_two: Coordinates | Tuple[float, ...],
    resolution: float,
) -> List[Coordinates | Tuple[float, ...]]:
    """A function that takes two consecutive points from either an exterior
    ring of a `shapely.geometry.Polygon` object or the coordinates of a
    `LineString` object and places equally spaced points along that line
    determined by the supplied geographic resolution. That resolution will
    be determined by the gridded input data.

    The resulting points will be appended to the rest of the ring,
    ensuring the ring has points at a resolution of the gridded data.

    """
    if len(point_one) < 2 or len(point_two) < 2:
        raise InvalidInputGeoJSON  # added to avoid type checking failures

    distance = np.linalg.norm(np.array(point_two[:2]) - np.array(point_one[:2]))
    n_points = np.ceil(distance / resolution) + 1
    new_x = np.linspace(point_one[0], point_two[0], int(n_points))
    new_y = np.linspace(point_one[1], point_two[1], int(n_points))

    return list(zip(new_x, new_y))


def get_x_y_extents_from_geographic_perimeter(
    points: List[Coordinates],
    crs: CRS,
    source_granule_extent: dict[str, float],
) -> Dict[str, float]:
    """Take an input list of (longitude, latitude) coordinates that define the
    exterior of the input GeoJSON shape or bounding box, and project those
    points to the target grid. Then return the minimum and maximum values
    of those projected coordinates. Check first for perimeter exceeding grid on
    all axes (whole grid extents returned). Then remove any points that are
    outside the grid before finding the min and max extent.

    """
    # get the x,y projected values from the geographic points
    point_longitudes, point_latitudes = zip(*points)
    from_geo_transformer = Transformer.from_crs(4326, crs)
    points_x, points_y = (  # pylint: disable=unpacking-non-sequence
        from_geo_transformer.transform(point_latitudes, point_longitudes)
    )

    # Filter out where projection is NaN or Inf
    finite_x, finite_y = remove_non_finite_projected_values(points_x, points_y)

    # Check if perimeter exceeds the grid extents on all axes. If true, return
    # whole grid extents and skips the code that follows (which fails in
    # this case).
    if perimeter_surrounds_grid(finite_x, finite_y, source_granule_extent):
        return source_granule_extent

    # Remove any points that are outside the grid
    finite_x, finite_y = remove_points_outside_grid_extents(
        finite_x, finite_y, source_granule_extent
    )

    return {
        "x_min": np.min(finite_x),
        "x_max": np.max(finite_x),
        "y_min": np.min(finite_y),
        "y_max": np.max(finite_y),
    }


def remove_non_finite_projected_values(
    points_x: np.ndarray, points_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Removes any NaN and infinity values and returns the results as numpy arrays"""
    # isfinite checks for NaN and infinty values returned for certain projections
    points_x = np.asarray(points_x)
    points_y = np.asarray(points_y)

    finite_mask = np.isfinite(points_x) & np.isfinite(points_y)
    finite_x = points_x[finite_mask]
    finite_y = points_y[finite_mask]
    return finite_x, finite_y


def perimeter_surrounds_grid(
    finite_x: np.ndarray, finite_y: np.ndarray, granule_extent: dict[str, float]
) -> bool:
    """Returns True if perimeter exceeds the grid extents on all axes.
    Returns False if does not.
    """
    if (
        np.min(finite_x) < granule_extent["x_min"]
        and np.max(finite_x) > granule_extent["x_max"]
        and np.min(finite_y) < granule_extent["y_min"]
        and np.max(finite_y) > granule_extent["y_max"]
    ):
        return True

    return False


def remove_points_outside_grid_extents(
    finite_x: np.ndarray,
    finite_y: np.ndarray,
    granule_extent: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Remove any points that are outside the grid and are invalid and raise an
    exception if the resulting grid is empty.

    """
    tolerance = 1e-9
    # This gets the mask of points within the granule extent.
    # The points are checked to make sure they are within
    # all 4 extents

    mask = (
        (finite_x >= granule_extent["x_min"] - tolerance)
        & (finite_x <= granule_extent["x_max"] + tolerance)
        & (finite_y >= granule_extent["y_min"] - tolerance)
        & (finite_y <= granule_extent["y_max"] + tolerance)
    )

    finite_x = finite_x[mask]
    finite_y = finite_y[mask]

    if finite_x.size == 0 or finite_y.size == 0:
        raise InvalidRequestedRange

    return finite_x, finite_y
