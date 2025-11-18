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
from varinfo import VarInfoFromDmr

from hoss.bbox_utilities import BBox, flatten_list
from hoss.exceptions import (
    InvalidGranuleDimensions,
    InvalidInputGeoJSON,
    InvalidRequestedRange,
    MissingGridMappingMetadata,
    MissingGridMappingVariable,
    MissingSpatialSubsetInformation,
)

Coordinates = Tuple[float]
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
    grid_mapping = next(
        iter(varinfo.get_variable(variable).references.get('grid_mapping', [])), None
    )

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
    doesn't exist, a `None` value will be returned.

    """
    return get_grid_mapping_attributes(variable, varinfo).get(
        "master_geotransform", None
    )


def get_projected_x_y_variables(
    varinfo: VarInfoFromDmr, variable: str
) -> Tuple[Optional[str]]:
    """Retrieve the names of the projected x and y dimension variables
    associated with a variable. If either are not found, a `None` value
    will be returned for the absent dimension variable.

    Note - the input variables to this function are only filtered to remove
    variables that are spatial dimensions. The input to this function may
    have no dimensions, or may not be spatially gridded.

    """
    variable_dimensions = varinfo.get_variable(variable).dimensions

    projected_x = next(
        (
            dimension
            for dimension in variable_dimensions
            if is_projection_x_dimension(varinfo, dimension)
        ),
        None,
    )

    projected_y = next(
        (
            dimension
            for dimension in variable_dimensions
            if is_projection_y_dimension(varinfo, dimension)
        ),
        None,
    )

    return projected_x, projected_y


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
    projected_x_names = ('projection_x_coordinate', 'projection_x_angular_coordinate')

    return varinfo.get_variable(dimension_variable) is not None and (
        varinfo.get_variable(dimension_variable).attributes.get('standard_name')
        in projected_x_names
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
    projected_y_names = ('projection_y_coordinate', 'projection_y_angular_coordinate')

    return varinfo.get_variable(dimension_variable) is not None and (
        varinfo.get_variable(dimension_variable).attributes.get('standard_name')
        in projected_y_names
    )


def get_projected_x_y_extents(
    x_values: np.ndarray,
    y_values: np.ndarray,
    crs: CRS,
    shape_file: str = None,
    bounding_box: BBox = None,
) -> Dict[str, float]:
    """Retrieve the minimum and maximum values for a projected grid as derived
    from either a bounding box or GeoJSON shape file, both of which are
    defined in geographic coordinates.

    A minimum grid resolution will be determined in the geographic
    Coordinate Reference System (CRS). The input spatial constraint will
    then have points populated around its exterior at this resolution.
    These geographic points will then all be projected to the target grid
    CRS, allowing the retrieval of a minimum and maximum value in both the
    projected x and projected y dimension.

    Example output:

    x_y_extents = {'x_min': 1000,
                   'x_max': 4000,
                   'y_min': 2500,
                   'y_max': 5500}

    """
    grid_lats, grid_lons = get_grid_lat_lons(  # pylint: disable=unpacking-non-sequence
        x_values, y_values, crs
    )
    if not np.all(np.isfinite(grid_lats)) or not np.all(np.isfinite(grid_lons)):
        raise InvalidGranuleDimensions

    # When projected, the perimeter of a bounding box or polygon in geographic
    # terms will become curved.  To determine the X, Y extent of the requested
    # bounding area, we need a perimeter with a suitable density of points, such that
    # we catch that curve when projected. The source file resolution is used to define
    # the necessary number of points (density).
    geographic_resolution = get_geographic_resolution(grid_lons, grid_lats)

    densified_perimeter = get_densified_perimeter(
        geographic_resolution, shape_file=shape_file, bounding_box=bounding_box
    )

    # To avoid out-of-limits projection, we need to clip the bounding perimeter to
    # the source file's geographic extents
    granule_extent = BBox(
        np.min(grid_lons), np.min(grid_lats), np.max(grid_lons), np.max(grid_lats)
    )

    clipped_perimeter = get_filtered_points(densified_perimeter, granule_extent)

    granule_extent_projected_meters = {
        "x_min": np.min(x_values),
        "x_max": np.max(x_values),
        "y_min": np.min(y_values),
        "y_max": np.max(y_values),
    }
    return get_x_y_extents_from_geographic_perimeter(
        clipped_perimeter, crs, granule_extent_projected_meters
    )


def get_filtered_points(
    points_in_requested_extent: List[Coordinates], granule_extent: BBox
) -> List[Coordinates]:
    """Returns lat/lon values clipped to the extent of the granule"""

    requested_lons, requested_lats = zip(*points_in_requested_extent)

    # all lon values are clipped within the granule lon extent
    clipped_lons = np.clip(requested_lons, granule_extent.west, granule_extent.east)

    # all lat values are clipped to granule lat extent
    clipped_lats = np.clip(requested_lats, granule_extent.south, granule_extent.north)

    return list(zip(clipped_lons, clipped_lats))


def get_grid_lat_lons(
    x_values: np.ndarray, y_values: np.ndarray, crs: CRS
) -> Tuple[np.ndarray]:
    """Construct a 2-D grid of projected x and y values from values in the
    corresponding dimension variable 1-D arrays. Then transform those
    points to longitudes and latitudes.

    """
    projected_x = np.repeat(x_values.reshape(1, len(x_values)), len(y_values), axis=0)
    projected_y = np.repeat(y_values.reshape(len(y_values), 1), len(x_values), axis=1)

    to_geo_transformer = Transformer.from_crs(crs, 4326)

    return to_geo_transformer.transform(  # pylint: disable=unpacking-non-sequence
        projected_x, projected_y
    )


def get_geographic_resolution(longitudes: np.ndarray, latitudes: np.ndarray) -> float:
    """Calculate the distance between diagonally adjacent cells in both
    longitude and latitude. Combined those differences in quadrature to
    obtain Euclidean distances. Return the minimum of these Euclidean
    distances. Over the typical distances being considered, differences
    between the Euclidean and geodesic distance between points should be
    minimal, with Euclidean distances being slightly shorter.

    """
    lon_square_diffs = np.square(np.subtract(longitudes[1:, 1:], longitudes[:-1, :-1]))
    lat_square_diffs = np.square(np.subtract(latitudes[1:, 1:], latitudes[:-1, :-1]))
    return np.nanmin(np.sqrt(np.add(lon_square_diffs, lat_square_diffs)))


def get_densified_perimeter(
    resolution: float, shape_file: str = None, bounding_box: BBox = None
) -> List[Coordinates]:
    """Take a shape file or bounding box, as defined by the input Harmony
    request, and return a full set of points that correspond to the
    exterior of any GeoJSON shape fixed to the resolution of the projected
    grid of the data.

    """
    if bounding_box is not None:
        resolved_geojson = get_resolved_feature(
            get_bbox_polygon(bounding_box), resolution
        )
    elif shape_file is not None:
        with open(shape_file, 'r', encoding='utf-8') as file_handler:
            geojson_content = json.load(file_handler)

        resolved_geojson = get_resolved_features(geojson_content, resolution)
    else:
        raise MissingSpatialSubsetInformation()

    return resolved_geojson


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
    """Parse GeoJSON read from a file. Once `shapely.geometry.shape` objects
    have been created for all features, these features will be resolved
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
        'geometrycollection',
        'linestring',
        'point',
        'polygon',
        'multilinestring',
        'multipoint',
        'multipolygon',
    )

    if geojson_content.get('type', '').lower() in feature_types:
        resolved_features = get_resolved_feature(shape(geojson_content), resolution)
    elif 'geometry' in geojson_content:
        resolved_features = get_resolved_feature(
            shape(geojson_content['geometry']), resolution
        )
    elif 'features' in geojson_content:
        resolved_features = flatten_list(
            [
                get_resolved_feature(shape(feature['geometry']), resolution)
                for feature in geojson_content['features']
            ]
        )
    else:
        raise InvalidInputGeoJSON()

    return resolved_features


def get_resolved_feature(feature: Shape, resolution: float) -> List[Coordinates]:
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
                for sub_geometry in feature.geoms
            ]
        )
    else:
        raise InvalidInputGeoJSON()

    return resolved_points


def get_resolved_geometry(
    geometry_points: List[Coordinates], resolution: float, is_closed: bool = True
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
    point_one: Coordinates, point_two: Coordinates, resolution: float
) -> List[Coordinates]:
    """A function that takes two consecutive points from either an exterior
    ring of a `shapely.geometry.Polygon` object or the coordinates of a
    `LineString` object and places equally spaced points along that line
    determined by the supplied geographic resolution. That resolution will
    be determined by the gridded input data.

    The resulting points will be appended to the rest of the ring,
    ensuring the ring has points at a resolution of the gridded data.

    """
    distance = np.linalg.norm(np.array(point_two[:2]) - np.array(point_one[:2]))
    n_points = np.ceil(distance / resolution) + 1
    new_x = np.linspace(point_one[0], point_two[0], int(n_points))
    new_y = np.linspace(point_one[1], point_two[1], int(n_points))
    return list(zip(new_x, new_y))


def get_x_y_extents_from_geographic_perimeter(
    points: List[Coordinates],
    crs: CRS,
    granule_extent_projected_meters: dict[str, float],
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

    finite_x, finite_y = remove_non_finite_projected_values(points_x, points_y)

    # Check if perimeter exceeds the grid extents on all axes. If true, return
    # whole grid extents and skips the code that follows (which fails in
    # this case).
    if perimeter_surrounds_grid(finite_x, finite_y, granule_extent_projected_meters):
        return granule_extent_projected_meters

    # Remove any points that are outside the grid
    finite_x, finite_y = remove_points_outside_grid_extents(
        finite_x, finite_y, granule_extent_projected_meters
    )

    return {
        'x_min': np.min(finite_x),
        'x_max': np.max(finite_x),
        'y_min': np.min(finite_y),
        'y_max': np.max(finite_y),
    }


def remove_non_finite_projected_values(
    points_x: list[float], points_y: list[float]
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
    finite_x: np.ndarray, finite_y: np.ndarray, granule_extent: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Remove any points that are outside the grid and are invalid and raise an
    exception if the resulting grid is empty.

    """
    tolerance = 1e-9
    # This gets the mask of points within the granule extent.
    # The individual extents are checked to make sure the points are within 
    # all 4 extents    
    mask = np.logical_and.reduce(
        [
            # points >= x_min allowing for a tolerance level
            np.logical_or(
                finite_x > granule_extent["x_min"],
                np.isclose(finite_x, granule_extent["x_min"], atol=tolerance),
            ),
            # points <= x_max allowing for a tolerance level
            np.logical_or(
                finite_x < granule_extent["x_max"],
                np.isclose(finite_x, granule_extent["x_max"], atol=tolerance),
            ),
            # points >= y_min allowing for a tolerance level
            np.logical_or(
                finite_y > granule_extent["y_min"],
                np.isclose(finite_y, granule_extent["y_min"], atol=tolerance),
            ),
            # points <= y_max allowing for a tolerance level
            np.logical_or(
                finite_y < granule_extent["y_max"],
                np.isclose(finite_y, granule_extent["y_max"], atol=tolerance),
            ),
        ]
    )

    finite_x = finite_x[mask]
    finite_y = finite_y[mask]

    if finite_x.size == 0 or finite_y.size == 0:
        raise InvalidRequestedRange

    return finite_x, finite_y
