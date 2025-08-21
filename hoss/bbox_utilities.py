"""This module contains utility functions relating to bounding box
calculations. This includes deriving a bounding box from an input GeoJSON
shape file.

Some of the functions in this module were written instead of using
[the shapely Python package](https://shapely.readthedocs.io/en/latest/),
as `shapely` does not handle antimeridian crossing when determining an
encompassing bounding box of a GeoJSON input. The
[GeoJSON specification](https://datatracker.ietf.org/doc/html/rfc7946#section-3.1.3)
encourages users to split GeoJSON at the antimeridian, however, this is not
enforced. When a user specifies a GeoJSON shape that crosses the
antimeridian, HOSS will use an encompassing bounding box that also crosses
the antimeridian.

"""

import json
from collections import namedtuple
from logging import Logger
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from harmony_service_lib.message import Message
from harmony_service_lib.util import Config, download

from hoss.exceptions import InvalidInputGeoJSON, UnsupportedShapeFileFormat

AggCoordinates = List[Tuple[float]]
BBox = namedtuple('BBox', ['west', 'south', 'east', 'north'])
Coordinates = Union[
    List[float],
    List[List[float]],
    List[List[List[float]]],
    List[List[List[List[float]]]],
]
GeoJSON = Union[Dict, List]


def get_harmony_message_bbox(message: Message) -> Optional[BBox]:
    """Try to retrieve a bounding box from an input Harmony message. If there
    is no bounding box, return None.

    """
    if message.subset is not None and message.subset.bbox is not None:
        bounding_box = BBox(*message.subset.process('bbox'))
    else:
        bounding_box = None

    return bounding_box


def get_request_shape_file(
    message: Message, working_dir: str, adapter_logger: Logger, adapter_config: Config
) -> str:
    """This helper function downloads the file specified in the input Harmony
    message via: `Message.subset.shape.href` and returns the local file
    path.

    """
    if message.subset is not None and message.subset.shape is not None:
        if message.subset.shape.type != 'application/geo+json':
            raise UnsupportedShapeFileFormat(message.subset.shape.type)

        shape_file_url = message.subset.shape.process('href')
        adapter_logger.info('Downloading request shape file')
        local_shape_file_path = download(
            shape_file_url,
            working_dir,
            logger=adapter_logger,
            access_token=message.accessToken,
            cfg=adapter_config,
        )
    else:
        local_shape_file_path = None

    return local_shape_file_path


def get_shape_file_geojson(local_shape_file_path: str) -> GeoJSON:
    """Retrieve the shape file GeoJSON from the downloaded shape file provided
    by the Harmony request.

    """
    with open(local_shape_file_path, 'r', encoding='utf-8') as file_handler:
        geojson_content = json.load(file_handler)

    return geojson_content


def get_geographic_bbox(geojson_input: GeoJSON) -> Optional[BBox]:
    """This function takes a GeoJSON input and extracts the longitudinal and
    latitudinal extents from it. These extents describe a bounding box that
    minimally encompasses the specified shape.

    This function should be used in cases where the data within the granule
    are geographic. Some projections, particularly polar projections, will
    require further refinement of the GeoJSON shape.

    In the function below `contiguous_bboxes` and `contiguous_bbox` refer
    to bounding boxes that do not cross the antimeridian. Although, the
    GeoJSON specification recommends that GeoJSON shapes should be split to
    avoid crossing the antimeridian, user-supplied shape files may not
    conform to this recommendation.

    """
    if 'bbox' in geojson_input:
        return get_bounding_box_lon_lat(geojson_input['bbox'])

    grouped_coordinates = aggregate_all_geometries(geojson_input)

    if len(grouped_coordinates) == 0:
        return None

    contiguous_bbox = get_contiguous_bbox(grouped_coordinates)
    antimeridian_bbox = get_antimeridian_bbox(grouped_coordinates)

    bbox_south, bbox_north = get_latitude_range(contiguous_bbox, antimeridian_bbox)

    if antimeridian_bbox is None:
        bbox_west = contiguous_bbox.west
        bbox_east = contiguous_bbox.east
    elif contiguous_bbox is None:
        bbox_west = antimeridian_bbox.west
        bbox_east = antimeridian_bbox.east
    elif bbox_in_longitude_range(
        contiguous_bbox, -180, antimeridian_bbox.east
    ) or bbox_in_longitude_range(contiguous_bbox, antimeridian_bbox.west, 180):
        # Antimeridian bounding box encompasses non-antimeridian crossing
        # bounding box
        bbox_west = antimeridian_bbox.west
        bbox_east = antimeridian_bbox.east
    elif (antimeridian_bbox.east - contiguous_bbox.west) < (
        contiguous_bbox.east - antimeridian_bbox.west
    ):
        # Distance from contiguous bounding box west to antimeridian bounding
        # box east is shorter than antimeridian bounding box west to contiguous
        # bounding box east
        bbox_west = contiguous_bbox.west
        bbox_east = antimeridian_bbox.east
    else:
        # Distance from antimeridian bounding box west to contiguous bounding
        # box east is shorter than contiguous bounding box west to antimeridian
        # bounding box east
        bbox_west = antimeridian_bbox.west
        bbox_east = contiguous_bbox.east

    return BBox(bbox_west, bbox_south, bbox_east, bbox_north)


def get_contiguous_bbox(grouped_coordinates: List[AggCoordinates]) -> Optional[BBox]:
    """Retrieve a bounding box that encapsulates all shape file geometries
    that do not cross the antimeridian.

    """
    contiguous_bboxes = [
        [min(grouped_lons), min(grouped_lats), max(grouped_lons), max(grouped_lats)]
        for grouped_lons, grouped_lats in grouped_coordinates
        if len(grouped_lons) == 1 or not crosses_antimeridian(grouped_lons)
    ]

    if len(contiguous_bboxes) > 0:
        aggregated_extents = list(zip(*contiguous_bboxes))
        contiguous_bbox = BBox(
            min(aggregated_extents[0]),
            min(aggregated_extents[1]),
            max(aggregated_extents[2]),
            max(aggregated_extents[3]),
        )
    else:
        contiguous_bbox = None

    return contiguous_bbox


def get_antimeridian_bbox(grouped_coordinates: List[AggCoordinates]) -> Optional[BBox]:
    """Retrieve a bounding box that encapsulates all shape file geometries
    that cross the antimeridian. The output bounding box will also cross
    the antimeridian.

    """
    antimeridian_bboxes = [
        get_antimeridian_geometry_bbox(grouped_lons, grouped_lats)
        for grouped_lons, grouped_lats in grouped_coordinates
        if len(grouped_lons) > 1 and crosses_antimeridian(grouped_lons)
    ]

    if len(antimeridian_bboxes) > 0:
        aggregated_extents = list(zip(*antimeridian_bboxes))
        antimeridian_bbox = BBox(
            min(aggregated_extents[0]),
            min(aggregated_extents[1]),
            max(aggregated_extents[2]),
            max(aggregated_extents[3]),
        )
    else:
        antimeridian_bbox = None

    return antimeridian_bbox


def get_antimeridian_geometry_bbox(
    grouped_lons: Tuple[float], grouped_lats: Tuple[float]
) -> BBox:
    """Combine the longitudes and latitudes for a single GeoJSON geometry into
    a bounding box that encapsulates that geometry. The input to this
    function will already have been identified as crossing the
    antimeridian. The longitudes will be split into two groups either side
    of the antimeridian, so the westernmost point west of the antimeridian
    and the easternmost point east of the antimeridian can be found.

    This function assumes that, on average, those points east of the
    antimeridian will have a lower average longitude than those west of it.

    The output from this function will be a bounding box that also crosses
    the antimeridian.

    """
    longitudes_group_one = [grouped_lons[0]]
    longitudes_group_two = []
    current_group = longitudes_group_one

    for previous_index, longitude in enumerate(grouped_lons[1:]):
        if crosses_antimeridian([longitude, grouped_lons[previous_index]]):
            if current_group == longitudes_group_one:
                current_group = longitudes_group_two
            else:
                current_group = longitudes_group_one

        current_group.append(longitude)

    if np.mean(longitudes_group_one) < np.mean(longitudes_group_two):
        east_lons = longitudes_group_one
        west_lons = longitudes_group_two
    else:
        east_lons = longitudes_group_two
        west_lons = longitudes_group_one

    return BBox(min(west_lons), min(grouped_lats), max(east_lons), max(grouped_lats))


def get_latitude_range(
    contiguous_bbox: Optional[BBox], antimeridian_bbox: Optional[BBox]
) -> Tuple[float]:
    """Retrieve the southern and northern extent for all bounding boxes. One
    of `contiguous_bbox` or `antimeridian_bbox` must not be `None`.

    * `contiguous_bbox`: A bounding box that minimally encompasses all
      GeoJSON geometries that do not cross the antimeridian.
    * `antimeridian_bbox`: A bounding box that minimally encompasses all
      GeoJSON geometries that _do_ cross the antimeridian.

    """
    south_values = [
        bbox.south for bbox in [contiguous_bbox, antimeridian_bbox] if bbox is not None
    ]
    north_values = [
        bbox.north for bbox in [contiguous_bbox, antimeridian_bbox] if bbox is not None
    ]

    return min(south_values), max(north_values)


def bbox_in_longitude_range(
    bounding_box: BBox, west_limit: float, east_limit: float
) -> bool:
    """Check if the specified bounding box is entirely contained by the
    specified longitude range.

    This function is used to identify when geometries that do not cross the
    antimeridian are contained by the longitudinal range of those that do.

    """
    return (
        west_limit <= bounding_box[0] <= east_limit
        and west_limit <= bounding_box[2] <= east_limit
    )


def aggregate_all_geometries(geojson_input: GeoJSON) -> List[AggCoordinates]:
    """Parse the input GeoJSON object, and identify all items within it
    containing geometries. When items containing geometries are identified,
    functions are called to aggregate the coordinates within each geometry
    and return a list of aggregated longitudes and latitudes for each
    geometry (or sub-geometry member, e.g., multiple points, linestrings or
    polygons).

    """
    if 'coordinates' in geojson_input:
        # A Geometry object with a `coordinates` attribute, e.g., Point,
        # LineString, Polygon, etc.
        grouped_coords = aggregate_geometry_coordinates(geojson_input['coordinates'])
    elif 'geometries' in geojson_input:
        # A GeometryCollection geometry.
        grouped_coords = flatten_list(
            [
                aggregate_geometry_coordinates(geometry['coordinates'])
                for geometry in geojson_input['geometries']
            ]
        )
    elif 'geometry' in geojson_input and 'coordinates' in geojson_input['geometry']:
        # A GeoJSON Feature (e.g., Point, LineString, Polygon, etc)
        grouped_coords = aggregate_geometry_coordinates(
            geojson_input['geometry']['coordinates']
        )
    elif 'geometry' in geojson_input and 'geometries' in geojson_input['geometry']:
        # A GeoJSON Feature containing a GeometryCollection
        grouped_coords = flatten_list(
            [
                aggregate_all_geometries(geometry)
                for geometry in geojson_input['geometry']['geometries']
            ]
        )
    elif 'features' in geojson_input:
        # A GeoJSON FeatureCollection
        grouped_coords = flatten_list(
            aggregate_all_geometries(feature) for feature in geojson_input['features']
        )
    else:
        raise InvalidInputGeoJSON()

    return grouped_coords


def aggregate_geometry_coordinates(
    coordinates: Coordinates, aggregated_coordinates: List[AggCoordinates] = None
) -> List[AggCoordinates]:
    """Extract the aggregated latitude and longitude coordinates associated
    with all child items in the `coordinates` attribute of a GeoJSON
    geometry. The order of longitudes and latitudes are preserved to allow
    later checking for antimeridian crossing.

    Some geometries have multiple parts, such as MultiLineStrings or
    MultiPolygons. These each have their own entries in the output list,
    so that the bounding box of each can be derived independently. Keeping
    sub-geometries separate is important to avoid spurious identification
    of antimeridian crossing.

    Return value:

    [
        [(x_0, ..., x_M), (y_0, ..., y_M)],  # For GeoJSON sub-geometry one
        [(x_0, ..., x_N), (y_0, ..., y_N)]   # For GeoJSON sub-geometry two
    ]

    For geometry types: Point, LineString and Polygon, there will be only
    a single sub-geometry item in the returned list.

    """
    if aggregated_coordinates is None:
        aggregated_coordinates = []

    if is_single_point(coordinates):
        aggregated_coordinates.append([(coordinates[0],), (coordinates[1],)])
    elif is_list_of_coordinates(coordinates):
        aggregated_coordinates.append(list(zip(*coordinates)))
    else:
        for nested_coordinates in coordinates:
            aggregate_geometry_coordinates(nested_coordinates, aggregated_coordinates)

    return aggregated_coordinates


def is_list_of_coordinates(input_object) -> bool:
    """Checks if the input contains a list of coordinates, which Python will
    represent as a list of lists of numerical values, e.g.:

    ```Python
    list_of_coordinates = [[0.1, 0.2], [0.3, 0.4]]
    ```

    """
    return isinstance(input_object, list) and all(
        is_single_point(element) for element in input_object
    )


def is_single_point(input_object) -> bool:
    """Checks if the input is a single list of numbers. Note, coordinates may
    or may not include a vertical coordinate as a third element.

    """
    return (
        isinstance(input_object, list)
        and len(input_object) in (2, 3)
        and all(isinstance(element, (float, int)) for element in input_object)
    )


def flatten_list(list_of_lists: List[List]) -> List:
    """Flatten the top level of a list of lists, to combine all elements in
    the child lists to be child elements at the top level of the object.
    For example:

    Input: [[1, 2, 3], [4, 5, 6]]
    Output: [1, 2, 3, 4, 5, 6]

    """
    return [item for sub_list in list_of_lists for item in sub_list]


def crosses_antimeridian(
    longitudes: List[Union[float, int]], longitude_threshold: float = 180.0
) -> bool:
    """Check if a specified list of ordered longitudes crosses the
    antimeridian (+/- 180 degrees east). This check assumes that any points
    that are separated by more than 180 degrees east in longitude will
    cross the antimeridian. There are edge-cases where this may not be
    true, but it is a common condition used in similar checks:

    https://towardsdatascience.com/around-the-world-in-80-lines-crossing-the-antimeridian-with-python-and-shapely-c87c9b6e1513

    """
    return np.abs(np.diff(longitudes)).max() > longitude_threshold


def get_bounding_box_lon_lat(bounding_box: List[float]) -> BBox:
    """Parse a GeoJSON bounding box attribute, and retrieve only the
    horizontal coordinates (West, South, East, North).

    """
    if len(bounding_box) == 4:
        horizontal_bounding_box = BBox(*bounding_box)
    elif len(bounding_box) == 6:
        horizontal_bounding_box = BBox(
            bounding_box[0], bounding_box[1], bounding_box[3], bounding_box[4]
        )
    else:
        raise InvalidInputGeoJSON()

    return horizontal_bounding_box
