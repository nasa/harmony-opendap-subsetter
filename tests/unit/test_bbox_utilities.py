""" This module tests the functions included in the `pymods/bbox_utilities.py`
    module. Those functions focus on retrieving bounding box information from
    a Harmony message, either from the `Message.subset.bbox` attribute or the
    `Message.subset.shape` attribute. In the case of a shape file, the module
    has code to convert a GeoJSON shape into a bounding box that minimally
    encloses the shape.

"""
from logging import getLogger
from os.path import join as path_join
from unittest import TestCase
from unittest.mock import patch
import json

from harmony.message import Message
from harmony.util import config

from pymods.bbox_utilities import (aggregate_all_geometries,
                                   aggregate_geometry_coordinates, BBox,
                                   bbox_in_longitude_range,
                                   crosses_antimeridian,
                                   flatten_list,
                                   get_bounding_box_lon_lat,
                                   get_antimeridian_bbox,
                                   get_antimeridian_geometry_bbox,
                                   get_contiguous_bbox,
                                   get_geographic_bbox,
                                   get_harmony_message_bbox,
                                   get_latitude_range,
                                   get_request_shape_file,
                                   get_shape_file_geojson,
                                   is_list_of_coordinates, is_single_point)
from pymods.exceptions import InvalidInputGeoJSON, UnsupportedShapeFileFormat


class TestBBoxUtilities(TestCase):
    """ A class for testing functions in the `pymods.bbox_utilities`
        module.

    """
    @classmethod
    def setUpClass(cls):
        cls.config = config(validate=False)
        cls.logger = getLogger('tests')
        cls.point_geojson = cls.read_geojson('point.geo.json')
        cls.multipoint_geojson = cls.read_geojson('multipoint.geo.json')
        cls.linestring_geojson = cls.read_geojson('linestring.geo.json')
        cls.multilinestring_geojson = cls.read_geojson('multilinestring.geo.json')
        cls.polygon_geojson = cls.read_geojson('polygon.geo.json')
        cls.multipolygon_geojson = cls.read_geojson('multipolygon.geo.json')
        cls.geometrycollection_geojson = cls.read_geojson('geometrycollection.geo.json')
        cls.multi_features_geojson = cls.read_geojson('multi_feature.geo.json')

    @staticmethod
    def read_geojson(geojson_basename: str):
        """ A helper function to extract GeoJSON from a supplied file path. """
        geojson_path = path_join('tests/geojson_examples', geojson_basename)

        with open(geojson_path, 'r') as file_handler:
            geojson_content = json.load(file_handler)

        return geojson_content

    def test_get_harmony_message_bbox(self):
        """ Ensure a BBox object is returned from an input Harmony message if
            there is a bounding box included in that message.

        """
        with self.subTest('There is a bounding box in the message.'):
            message = Message({'subset': {'bbox': [1, 2, 3, 4]}})
            self.assertTupleEqual(get_harmony_message_bbox(message),
                                  BBox(1, 2, 3, 4))

        with self.subTest('There is a shape file in the message, but no bbox'):
            message = Message({
                'subset': {'shape': {'href': 'www.example.com/shape.geo.json',
                                     'type': 'application/geo_json'}}
            })
            self.assertIsNone(get_harmony_message_bbox(message))

        with self.subTest('There is no subset attribute to the message.'):
            message = Message({'accessToken': '0p3n5354m3'})
            self.assertIsNone(get_harmony_message_bbox(message))

    def test_get_shape_file_geojson(self):
        """ Ensure that a local GeoJSON file is correctly read. """
        read_geojson = get_shape_file_geojson(
            'tests/geojson_examples/point.geo.json'
        )

        self.assertDictEqual(read_geojson, self.point_geojson)
        # Ensure that both files aren't just empty
        self.assertIn('features', read_geojson)

    @patch('pymods.bbox_utilities.download')
    def test_get_request_shape_file(self, mock_download):
        """ Ensure that a shape file is returned if present in an input Harmony
            message. If the shape file MIME type is incorrect, an exception
            should be raised. If no shape file is present, then the function
            should return None.

        """
        access_token = 'UUDDLRLRBA'
        local_dir = '/tmp'
        local_shape_file_path = '/tmp/local.geo.json'
        shape_file_url = 'shape.geo.json'

        mock_download.return_value = local_shape_file_path

        with self.subTest('Shape file provided'):
            message = Message({
                'accessToken': access_token,
                'subset': {'shape': {'href': shape_file_url,
                                     'type': 'application/geo+json'}}
            })

            self.assertEqual(get_request_shape_file(message, local_dir,
                                                    self.logger, self.config),
                             local_shape_file_path)

            mock_download.assert_called_once_with(shape_file_url, local_dir,
                                                  logger=self.logger,
                                                  access_token=access_token,
                                                  cfg=self.config)
        mock_download.reset_mock()

        with self.subTest('Shape file has wrong MIME type'):
            message = Message({
                'accessToken': access_token,
                'subset': {'shape': {'href': shape_file_url, 'type': 'bad'}}
            })

            with self.assertRaises(UnsupportedShapeFileFormat):
                get_request_shape_file(message, local_dir, self.logger,
                                       self.config)

                mock_download.assert_not_called()

        with self.subTest('No shape file in message'):
            message = Message({
                'accessToken': access_token,
                'subset': {'bbox': [10, 20, 30, 40]}
            })

            self.assertIsNone(get_request_shape_file(message, local_dir,
                                                     self.logger, self.config))

            mock_download.assert_not_called()

        with self.subTest('No subset property in message'):
            message = Message({'accessToken': access_token})

            self.assertIsNone(get_request_shape_file(message, local_dir,
                                                     self.logger, self.config))

            mock_download.assert_not_called()

    def test_get_geographic_bbox_antimeridian_combinations(self):
        """ Ensure that the correct bounding box is extracted for Features that
            cross the antimeridian:

            * An antimeridian crossing feature.
            * An antimeridian crossing feature and a nearby non-antimeridian
              crossing feature to the east (should extend the antimeridian
              bounding box eastwards to retrieve the least data).
            * An antimeridian crossing feature and a nearby non-antimeridian
              crossing feature to the west (should extend the antimeridian
              bounding box westwards to retrieve the least data).
            * An antimeridian crossing feature and a non-antimeridian crossing
              feature that lies entirely between the antimeridian and the
              western extent of the antimeridian crossing feature. The returned
              bounding box longitude extents should just be that of the
              antimeridian crossing feature.
            * An antimeridian crossing feature and a non-antimeridian crossing
              feature that lies entirely between the antimeridian and the
              eastern extent of the antimeridian crossing feature. The returned
              bounding box longitude extents should just be those of the
              antimeridian crossing feature.

        """
        test_args = [
            ['antimeridian_only.geo.json', BBox(175.0, 37.0, -176.0, 44.0)],
            ['antimeridian_west.geo.json', BBox(160.0, 37.0, -176.0, 55.0)],
            ['antimeridian_east.geo.json', BBox(175.0, 22.0, -160.0, 44.0)],
            ['antimeridian_within_west.geo.json',
             BBox(175.0, 37.0, -176.0, 44.0)],
            ['antimeridian_within_east.geo.json',
             BBox(175.0, 37.0, -176.0, 44.0)],
        ]

        for geojson_basename, expected_bounding_box in test_args:
            with self.subTest(geojson_basename):
                geojson = self.read_geojson(geojson_basename)
                self.assertTupleEqual(get_geographic_bbox(geojson),
                                      expected_bounding_box)

    @patch('pymods.bbox_utilities.aggregate_all_geometries')
    def test_get_geographic_bbox_geojson_has_bbox(self,
                                                  mock_aggregate_all_geometries):
        """ Ensure that, if present, the optional GeoJSON "bbox" attribute is
            used. This will mean that further parsing of the "coordinates" is
            not undertaken.

        """
        bbox_geojson = self.read_geojson('polygon_with_bbox.geo.json')

        self.assertTupleEqual(get_geographic_bbox(bbox_geojson),
                              BBox(-114.05, 37.0, -109.04, 42.0))

        # Because the bounding box was retrieved from the "bbox" attribute,
        # the function returns before it can call anything else.
        mock_aggregate_all_geometries.assert_not_called()

    def test_get_geographic_bbox_geojson_types(self):
        """ Ensure that the correct bounding box is extracted for Features of
            each of the core GeoJSON geometry types.

        """
        test_args = [
            ['Point', self.point_geojson, BBox(2.295, 48.874, 2.295, 48.874)],
            ['MultiPoint', self.multipoint_geojson,
             BBox(-0.142, 51.501, -0.076, 51.508)],
            ['LineString', self.linestring_geojson,
             BBox(-80.519, 38.471, -75.696, 39.724)],
            ['MultiLineString', self.multilinestring_geojson,
             BBox(-3.194, 51.502, -0.128, 55.953)],
            ['Polygon', self.polygon_geojson,
             BBox(-114.05, 37.0, -109.04, 42.0)],
            ['MultiPolygon', self.multipolygon_geojson,
             BBox(-111.05, 37.0, -102.05, 45.0)],
            ['GeometryCollection', self.geometrycollection_geojson,
             BBox(-80.519, 38.471, -75.565, 39.724)],
        ]

        for description, geojson, expected_bounding_box in test_args:
            with self.subTest(description):
                self.assertTupleEqual(get_geographic_bbox(geojson),
                                      expected_bounding_box)

    def test_get_contiguous_bbox(self):
        """ Ensure the aggregated longitudes and latitudes of one or more
            GeoJSON geometries that do not cross the antimeridian can be
            correctly combined to form a single bounding box.

        """
        # The input coordinates are aggregated:
        # [(lon_0, lon_1, ..., lon_N), (lat_0, lat_1, ..., lat_N)]
        point_coordinates = [(4, ), (6, )]
        linestring_coordinates = [(-10, 10), (-20, 20)]
        polygon_coordinates = [(30, 35, 35, 30, 30), (30, 30, 40, 40, 30)]

        with self.subTest('Point geometry'):
            self.assertTupleEqual(get_contiguous_bbox([point_coordinates]),
                                  BBox(4, 6, 4, 6))

        with self.subTest('Single geometry'):
            self.assertTupleEqual(
                get_contiguous_bbox([linestring_coordinates]),
                BBox(-10, -20, 10, 20)
            )

        with self.subTest('Multiple geometries'):
            self.assertTupleEqual(
                get_contiguous_bbox([linestring_coordinates, polygon_coordinates]),
                BBox(-10, -20, 35, 40)
            )

        with self.subTest('Feature crossing antimeridian returns None'):
            self.assertIsNone(get_contiguous_bbox([[(170, -170), (10, 20)]]))

    def test_get_antimeridian_bbox(self):
        """ Ensure the aggregated longitudes and latitudes of one or more
            GeoJSON geometries crossing the antimeridian can be correctly
            combined to form a single bounding box.

            Because these features cross the antimeridian, the bounding box
            will have a western extent that is greater than the eastern extent.

        """
        # The input coordinates are aggregated:
        # [(lon_0, lon_1, ..., lon_N), (lat_0, lat_1, ..., lat_N)]
        point_coordinates = [(0, ), (0, )]
        linestring_coordinates = [(160, -170), (-20, 20)]
        polygon_coordinates = [(165, -165, -165, 165, 165),
                               (30, 30, 40, 40, 30)]

        with self.subTest('Point returns None'):
            self.assertIsNone(get_antimeridian_bbox([point_coordinates]))

        with self.subTest('Input not crossing antimeridian returns None'):
            self.assertIsNone(get_antimeridian_bbox([[(10, 20), (10, 20)]]))

        with self.subTest('Single geometry'):
            self.assertTupleEqual(
                get_antimeridian_bbox([linestring_coordinates]),
                BBox(160, -20, -170, 20)
            )

        with self.subTest('Multiple geometries'):
            self.assertTupleEqual(
                get_antimeridian_bbox([linestring_coordinates,
                                       polygon_coordinates]),
                BBox(160, -20, -165, 40)
            )

    def test_get_antimeridian_geometry_bbox(self):
        """ Ensure the aggregated longitudes and latitudes of one or more
            GeoJSON geometries crossing the antimeridian can be correctly
            combined to form a single bounding box.

            Because these features cross the antimeridian, the bounding box
            will have a western extent that is greater than the eastern extent.

        """
        # The input coordinates are aggregated:
        # [(lon_0, lon_1, ..., lon_N), (lat_0, lat_1, ..., lat_N)]
        linestring_coordinates = [(160, -170), (-20, 20)]
        polygon_coordinates = [(165, -165, -165, 165, 165),
                               (30, 30, 40, 40, 30)]

        test_args = [
            ['LineString', linestring_coordinates, BBox(160, -20, -170, 20)],
            ['Polygon', polygon_coordinates, BBox(165, 30, -165, 40)]
        ]

        for description, coordinates, expected_bbox in test_args:
            with self.subTest(description):
                self.assertTupleEqual(
                    get_antimeridian_geometry_bbox(coordinates[0],
                                                   coordinates[1]),
                    expected_bbox
                )

    def test_get_latitude_range(self):
        """ Ensure that the broadest latitude range is extracted from a
            combination of those bounding boxes that cross the antimeridian
            and those that don't. The inputs to this function will include one
            or both of:

            * A bounding box encapsulating all GeoJSON features that do not
              cross the antimeridian.
            * A bounding box encapsulating all GeoJSON features that do cross
              the antimeridian.

        """
        antimeridian_bbox = BBox(170, -20, -170, 20)
        north_bbox = BBox(60, 30, 80, 50)
        south_bbox = BBox(60, -60, 80, -40)
        overlapping_bbox = BBox(60, 10, 80, 30)
        taller_bbox = BBox(60, -30, 80, 30)
        shorter_bbox = BBox(60, -10, 80, 10)

        test_args = [
            ['Contiguous bbox only', north_bbox, None, (30, 50)],
            ['Antimeridian bbox only', None, antimeridian_bbox, (-20, 20)],
            ['Contiguous north of antimeridian', north_bbox, antimeridian_bbox,
             (-20, 50)],
            ['Contiguous south of antimeridian', south_bbox, antimeridian_bbox,
             (-60, 20)],
            ['Overlapping bboxes', overlapping_bbox, antimeridian_bbox,
             (-20, 30)],
            ['Contiguous range contains antimeridian', taller_bbox,
             antimeridian_bbox, (-30, 30)],
            ['Contiguous range contained by antimeridian', shorter_bbox,
             antimeridian_bbox, (-20, 20)]
        ]

        for description, contiguous_bbox, am_bbox, expected_range in test_args:
            with self.subTest(description):
                self.assertTupleEqual(get_latitude_range(contiguous_bbox,
                                                         am_bbox),
                                      expected_range)

    def test_bbox_in_longitude_range(self):
        """ Ensure that the function correctly identifies when a bounding box
            lies entirely in the supplied longitude range.

        """
        bounding_box = BBox(30, 10, 40, 20)

        with self.subTest('Bounding box is in range'):
            self.assertTrue(bbox_in_longitude_range(bounding_box, 25, 45))

        with self.subTest('Bounding box entirely outside range'):
            self.assertFalse(bbox_in_longitude_range(bounding_box, 0, 20))

        with self.subTest('Bounding box partially inside range'):
            self.assertFalse(bbox_in_longitude_range(bounding_box, 25, 35))

    def test_aggregate_all_geometries(self):
        """ Ensure that GeoJSON objects can all be aggregated if:

            * Only coordinates are supplied in the input.
            * The input is a Geometry (e.g., Point, etc)
            * The input is a GeometryCollection type.
            * The input is a Feature.
            * The input is a Feature containing a GeometryCollection.
            * The input is a FeatureCollection.
            * The input is a FeatureCollection with multiple features.

        """
        point_output = [[(2.295, ), (48.874, )]]
        geometrycollection_output = [
            [(-75.565, ), (39.662, )],
            [(-75.696, -75.795, -80.519), (38.471, 39.716, 39.724)]
        ]

        with self.subTest('Point geometry'):
            self.assertListEqual(
                aggregate_all_geometries(
                    self.point_geojson['features'][0]['geometry']
                ),
                point_output
            )

        with self.subTest('GeometryCollection geometry'):
            self.assertListEqual(
                aggregate_all_geometries(
                    self.geometrycollection_geojson['features'][0]['geometry']
                ),
                geometrycollection_output
            )

        with self.subTest('Point Feature'):
            self.assertListEqual(
                aggregate_all_geometries(self.point_geojson['features'][0]),
                point_output
            )

        with self.subTest('GeometryCollection Feature'):
            self.assertListEqual(
                aggregate_all_geometries(
                    self.geometrycollection_geojson['features'][0]
                ),
                geometrycollection_output
            )

        with self.subTest('Point FeatureCollection'):
            self.assertListEqual(aggregate_all_geometries(self.point_geojson),
                                 point_output)

        with self.subTest('FeatureCollection with multiple Features'):
            # The features in multi_feature.geo.json match those in
            # geometrycollection.geo.json
            self.assertListEqual(
                aggregate_all_geometries(self.multi_features_geojson),
                geometrycollection_output
            )

        with self.subTest('Bad GeoJSON raises exception'):
            with self.assertRaises(InvalidInputGeoJSON):
                aggregate_all_geometries({'bad': 'input'})

    def test_aggregate_geometry_coordinates(self):
        """ Ensure that different types of GeoJSON objects (Point, LineString,
            Polygon, etc) can have their coordinates grouped from lists of
            [longitude, latitude (and possibly vertical)] points to ordered,
            separate lists of each coordinate type.

        """
        test_args = [
            ['Point', self.point_geojson, [[(2.295, ), (48.874, )]]],
            ['MultiPoint', self.multipoint_geojson, [[(-0.076, -0.142),
                                                      (51.508, 51.501)]]],
            ['LineString', self.linestring_geojson,
             [[(-75.696, -75.795, -80.519), (38.471, 39.716, 39.724)]]],
            ['MultiLineString', self.multilinestring_geojson,
             [[(-3.194, -3.181, -3.174), (55.949, 55.951, 55.953)],
              [(-0.140, -0.128), (51.502, 51.507)]]],
            ['Polygon', self.polygon_geojson,
             [[(-114.05, -114.05, -109.04, -109.04, -111.05, -111.05, -114.05),
               (42.0, 37.0, 37.0, 41.0, 41.0, 42.0, 42.0)]]],
            ['MultiPolygon', self.multipolygon_geojson,
             [[(-109.05, -109.05, -102.05, -102.05, -109.05),
               (41.0, 37.0, 37.0, 41.0, 41.0)],
              [(-111.05, -111.05, -104.05, -104.05, -111.05),
               (45.0, 41.0, 41.0, 45.0, 45.0)]]],
        ]

        for description, geojson, expected_output in test_args:
            with self.subTest(description):
                coordinates = geojson['features'][0]['geometry']['coordinates']
                self.assertListEqual(
                    aggregate_geometry_coordinates(coordinates),
                    expected_output
                )

    def test_is_list_of_coordinates(self):
        """ Ensure a list of coordiantes can be correctly recognised, and that
            other inputs are note incorrectly considered a list of coordinates.

        """
        test_args = [['List of horizontal coordinates', [[1, 2], [3, 4]]],
                     ['List of vertical coordinates', [[1, 2, 3], [4, 5, 6]]]]

        for description, test_input in test_args:
            with self.subTest(description):
                self.assertTrue(is_list_of_coordinates(test_input))

        test_args = [['Input is not a list', 1.0],
                     ['Input elements are not coordinates', [1, 2]],
                     ['Coordinates item has wrong number of elements', [[1]]],
                     ['Input is too nested', [[[1.0, 2.0]]]]]

        for description, test_input in test_args:
            with self.subTest(description):
                self.assertFalse(is_list_of_coordinates(test_input))

    def test_is_single_point(self):
        """ Ensure a single coordinate can be correctly recognised, and that
            other inputs are not incorrectly considered a coordinate pair.

        """
        test_args = [['Only horizontal coordinates', [-120.0, 20.0]],
                     ['Vertical coordinate included', [-120.0, 20.0, 300.0]]]

        for description, test_input in test_args:
            with self.subTest(description):
                self.assertTrue(is_single_point(test_input))

        test_args = [['Wrong number of list elements', [1.0]],
                     ['Input not a list', 1.0],
                     ['List contains a nested list', [[[-120.0, 20.0]]]]]

        for description, test_input in test_args:
            with self.subTest('A non coordinate type returns False'):
                self.assertFalse(is_single_point(test_input))

    def test_flatten_list(self):
        """ Ensure a list of lists is flattened by only one level. """
        self.assertListEqual(flatten_list([[1, 2], [3, 4], [5, 6]]),
                             [1, 2, 3, 4, 5, 6])

        self.assertListEqual(flatten_list([[[1, 2], [3, 4]], [[5, 6]]]),
                             [[1, 2], [3, 4], [5, 6]])

    def test_crosses_antimeridian(self):
        """ Ensure that antimeridian crossing is correctly identified from an
            ordered tuple of longitudes. Note, this relies on assuming a
            separation between consecutive points over a certain threshold
            indicates antimeridian crossing, which may not always be accurate.

        """
        with self.subTest('Longitudes do cross antimeridian.'):
            self.assertTrue(crosses_antimeridian((140, 175, -175, 140)))

        with self.subTest('Longitudes do not cross antimeridian.'):
            self.assertFalse(crosses_antimeridian((140, 175, 150, 140)))

    def test_get_bounding_bbox_lon_lat(self):
        """ Ensure the horizontal components of a GeoJSON bounding box
            attribute can be correctly extracted, whether that bounding box
            contains only horizontal coordinates or also vertical components.

        """
        expected_bounding_box = BBox(-10, -5, 10, 15)

        with self.subTest('Bounding box only has horizontal coordinates'):
            self.assertTupleEqual(get_bounding_box_lon_lat([-10, -5, 10, 15]),
                                  expected_bounding_box)

        with self.subTest('Bounding box also has vertical coordinates'):
            self.assertTupleEqual(
                get_bounding_box_lon_lat([-10, -5, 20, 10, 15, 30]),
                expected_bounding_box
            )

        with self.subTest('Incorrect format raises exception'):
            with self.assertRaises(InvalidInputGeoJSON):
                get_bounding_box_lon_lat([1, 2, 3])
