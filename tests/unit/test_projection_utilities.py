""" This module tests the functions included in the
    `hoss/projection_utilities.py` module. Those functions focus on using the
    spatial constraint information from an input Harmony message with
    collections that have projected grids.

"""

import json
from os.path import join as path_join
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import call, patch

import numpy as np
from pyproj import CRS
from shapely.geometry import Polygon, shape
from varinfo import VarInfoFromDmr

from hoss.bbox_utilities import BBox
from hoss.exceptions import (
    InvalidInputGeoJSON,
    MissingGridMappingMetadata,
    MissingGridMappingVariable,
    MissingSpatialSubsetInformation,
)
from hoss.projection_utilities import (
    get_bbox_polygon,
    get_geographic_resolution,
    get_grid_lat_lons,
    get_projected_x_y_extents,
    get_projected_x_y_variables,
    get_resolved_feature,
    get_resolved_features,
    get_resolved_geojson,
    get_resolved_geometry,
    get_resolved_line,
    get_variable_crs,
    get_x_y_extents_from_geographic_points,
    is_projection_x_dimension,
    is_projection_y_dimension,
)


class TestProjectionUtilities(TestCase):
    """A class for testing functions in the `hoss.projection_utilities`
    module.

    """

    @classmethod
    def setUpClass(cls):
        # Set up GeoJSON fixtures (both as raw GeoJSON and parsed shapely objects)
        cls.geometry_coll_geojson = cls.read_geojson('geometrycollection.geo.json')
        cls.geometry_coll = shape(cls.geometry_coll_geojson['features'][0]['geometry'])
        cls.linestring_geojson = cls.read_geojson('linestring.geo.json')
        cls.linestring = shape(cls.linestring_geojson['features'][0]['geometry'])
        cls.mlinestring_geojson = cls.read_geojson('multilinestring.geo.json')
        cls.multi_linestring = shape(cls.mlinestring_geojson['features'][0]['geometry'])
        cls.mpoint_geojson = cls.read_geojson('multipoint.geo.json')
        cls.multi_point = shape(cls.mpoint_geojson['features'][0]['geometry'])
        cls.mpolygon_geojson = cls.read_geojson('multipolygon.geo.json')
        cls.multi_polygon = shape(cls.mpolygon_geojson['features'][0]['geometry'])
        cls.point_geojson = cls.read_geojson('point.geo.json')
        cls.point = shape(cls.point_geojson['features'][0]['geometry'])
        cls.polygon_file_name = 'polygon.geo.json'
        cls.polygon_geojson = cls.read_geojson(cls.polygon_file_name)
        cls.polygon = shape(cls.polygon_geojson['features'][0]['geometry'])

    def setUp(self):
        self.temp_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.temp_dir)

    @staticmethod
    def read_geojson(geojson_base_name: str):
        """A helper function to extract GeoJSON from a supplied file path."""
        with open(f'tests/geojson_examples/{geojson_base_name}', 'r') as file_handler:
            geojson_content = json.load(file_handler)

        return geojson_content

    def test_get_variable_crs(self):
        """Ensure a `pyproj.CRS` object can be instantiated via the reference
        in a variable. Alternatively, if the `grid_mapping` attribute is
        absent, or erroneous, ensure the expected exceptions are raised.

        """
        sample_dmr = (
            '<Dataset xmlns="namespace_string">'
            '  <Dimension name="x" size="500" />'
            '  <Dimension name="y" size="500" />'
            '  <String name="crs">'
            '    <Attribute name="false_easting" type="Float64">'
            '      <Value>0.</Value>'
            '    </Attribute>'
            '    <Attribute name="false_northing" type="Float64">'
            '      <Value>0.</Value>'
            '    </Attribute>'
            '    <Attribute name="latitude_of_projection_origin" type="Float64">'
            '      <Value>40.</Value>'
            '    </Attribute>'
            '    <Attribute name="longitude_of_central_meridian" type="Float64">'
            '      <Value>-96.</Value>'
            '    </Attribute>'
            '    <Attribute name="standard_parallel" type="Float64">'
            '      <Value>50.</Value>'
            '      <Value>70.</Value>'
            '    </Attribute>'
            '    <Attribute name="long_name" type="String">'
            '      <Value>CRS definition</Value>'
            '    </Attribute>'
            '    <Attribute name="longitude_of_prime_meridian" type="Float64">'
            '      <Value>0.</Value>'
            '    </Attribute>'
            '    <Attribute name="semi_major_axis" type="Float64">'
            '      <Value>6378137.</Value>'
            '    </Attribute>'
            '    <Attribute name="inverse_flattening" type="Float64">'
            '      <Value>298.25722210100002</Value>'
            '    </Attribute>'
            '    <Attribute name="grid_mapping_name" type="String">'
            '      <Value>albers_conical_equal_area</Value>'
            '    </Attribute>'
            '  </String>'
            '  <Int16 name="variable_with_grid_mapping">'
            '    <Dim name="/x" />'
            '    <Dim name="/y" />'
            '    <Attribute name="grid_mapping" type="String">'
            '      <Value>crs</Value>'
            '    </Attribute>'
            '  </Int16>'
            '  <Int16 name="variable_without_grid_mapping">'
            '    <Dim name="/x" />'
            '    <Dim name="/y" />'
            '  </Int16>'
            '  <Int16 name="variable_with_bad_grid_mapping">'
            '    <Dim name="/x" />'
            '    <Dim name="/y" />'
            '    <Attribute name="grid_mapping" type="String">'
            '      <Value>non_existent_crs</Value>'
            '    </Attribute>'
            '  </Int16>'
            '</Dataset>'
        )

        dmr_path = path_join(self.temp_dir, 'grid_mapping.dmr.xml')

        with open(dmr_path, 'w', encoding='utf-8') as file_handler:
            file_handler.write(sample_dmr)

        varinfo = VarInfoFromDmr(dmr_path)

        expected_crs = CRS.from_cf(
            {
                'false_easting': 0.0,
                'false_northing': 0.0,
                'latitude_of_projection_origin': 40.0,
                'longitude_of_central_meridian': -96.0,
                'standard_parallel': [50.0, 70.0],
                'long_name': 'CRS definition',
                'longitude_of_prime_meridian': 0.0,
                'semi_major_axis': 6378137.0,
                'inverse_flattening': 298.25722210100002,
                'grid_mapping_name': 'albers_conical_equal_area',
            }
        )

        with self.subTest('Variable with "grid_mapping" gets expected CRS'):
            actual_crs = get_variable_crs('/variable_with_grid_mapping', varinfo)
            self.assertEqual(actual_crs, expected_crs)

        with self.subTest('Variable has no "grid_mapping" attribute'):
            with self.assertRaises(MissingGridMappingMetadata) as context:
                get_variable_crs('/variable_without_grid_mapping', varinfo)

            self.assertEqual(
                context.exception.message,
                'Projected variable "/variable_without_grid_mapping"'
                ' does not have an associated "grid_mapping" '
                'metadata attribute.',
            )

        with self.subTest('"grid_mapping" points to non-existent variable'):
            with self.assertRaises(MissingGridMappingVariable) as context:
                get_variable_crs('/variable_with_bad_grid_mapping', varinfo)

            self.assertEqual(
                context.exception.message,
                'Grid mapping variable "/non_existent_crs" '
                'referred to by variable '
                '"/variable_with_bad_grid_mapping" is not '
                'present in granule .dmr file.',
            )

        with self.subTest('grid_mapping override with json configuration'):
            smap_varinfo = VarInfoFromDmr(
                'tests/data/SC_SPL3SMP_008.dmr',
                'SPL3SMP',
                'hoss/hoss_config.json',
            )
            expected_crs = CRS.from_cf(
                {
                    'false_easting': 0.0,
                    'false_northing': 0.0,
                    'longitude_of_central_meridian': 0.0,
                    'standard_parallel': 30.0,
                    'grid_mapping_name': 'lambert_cylindrical_equal_area',
                }
            )
            actual_crs = get_variable_crs(
                '/Soil_Moisture_Retrieval_Data_AM/surface_flag', smap_varinfo
            )
            self.assertEqual(actual_crs, expected_crs)

    def test_get_projected_x_y_extents(self):
        """Ensure that the expected values for the x and y dimension extents
        are recovered for a known projected grid and requested input.

        The dimension values used below mimic one of the ABoVE TVPRM
        granules. Both the bounding box and the shape file used are
        identical shapes, just expressed either as a bounding box or a
        GeoJSON polygon. They should therefore return the same extents.

        """
        x_values = np.linspace(-3385020, -1255020, 72)
        y_values = np.linspace(4625000, 3575000, 36)
        crs = CRS.from_cf(
            {
                'false_easting': 0.0,
                'false_northing': 0.0,
                'latitude_of_projection_origin': 40.0,
                'longitude_of_central_meridian': -96.0,
                'standard_parallel': [50.0, 70.0],
                'long_name': 'CRS definition',
                'longitude_of_prime_meridian': 0.0,
                'semi_major_axis': 6378137.0,
                'inverse_flattening': 298.257222101,
                'grid_mapping_name': 'albers_conical_equal_area',
            }
        )

        bounding_box = BBox(-160, 68, -145, 70)
        polygon = {
            'type': 'Polygon',
            'coordinates': [
                [
                    (bounding_box.west, bounding_box.south),
                    (bounding_box.east, bounding_box.south),
                    (bounding_box.east, bounding_box.north),
                    (bounding_box.west, bounding_box.north),
                    (bounding_box.west, bounding_box.south),
                ]
            ],
        }
        polygon_path = path_join(self.temp_dir, 'bbox_poly.geo.json')

        with open(polygon_path, 'w', encoding='utf-8') as file_handler:
            json.dump(polygon, file_handler, indent=4)

        expected_output = {
            'x_min': -2273166.953240025,
            'x_max': -1709569.3224678137,
            'y_min': 3832621.3156695124,
            'y_max': 4425654.159834823,
        }

        with self.subTest('Bounding box input'):
            self.assertDictEqual(
                get_projected_x_y_extents(
                    x_values, y_values, crs, bounding_box=bounding_box
                ),
                expected_output,
            )

        with self.subTest('Shape file input'):
            self.assertDictEqual(
                get_projected_x_y_extents(
                    x_values, y_values, crs, shape_file=polygon_path
                ),
                expected_output,
            )

    def test_get_projected_x_y_variables(self):
        """Ensure that the `standard_name` metadata attribute can be parsed
        via `VarInfoFromDmr` for all dimenions of a specifed variable. If
        no dimensions have either an x or y coordinate, the corresponding
        return value should be `None`.

        """
        sample_dmr = (
            '<Dataset xmlns="namespace_string">'
            '  <Dimension name="x" size="500" />'
            '  <Dimension name="y" size="500" />'
            '  <Dimension name="lat" size="360" />'
            '  <Dimension name="lon" size="720" />'
            '  <Float64 name="x">'
            '    <Dim name="/x" />'
            '    <Attribute name="standard_name" type="String">'
            '      <Value>projection_x_coordinate</Value>'
            '    </Attribute>'
            '  </Float64>'
            '  <Float64 name="y">'
            '    <Dim name="/y" />'
            '    <Attribute name="standard_name" type="String">'
            '      <Value>projection_y_coordinate</Value>'
            '    </Attribute>'
            '  </Float64>'
            '  <Float64 name="lat">'
            '    <Dim name="/lat" />'
            '    <Attribute name="standard_name" type="String">'
            '      <Value>latitude</Value>'
            '    </Attribute>'
            '    <Attribute name="units" type="String">'
            '      <Value>degrees_north</Value>'
            '    </Attribute>'
            '  </Float64>'
            '  <Float64 name="lon">'
            '    <Dim name="/lon" />'
            '    <Attribute name="standard_name" type="String">'
            '      <Value>longitude</Value>'
            '    </Attribute>'
            '    <Attribute name="units" type="String">'
            '      <Value>degrees_east</Value>'
            '    </Attribute>'
            '  </Float64>'
            '  <String name="crs">'
            '  </String>'
            '  <Float64 name="variable_with_x_y_dims">'
            '    <Dim name="/x" />'
            '    <Dim name="/y" />'
            '  </Float64>'
            '  <Float64 name="variable_with_x_dim">'
            '    <Dim name="/x" />'
            '  </Float64>'
            '  <Float64 name="variable_with_y_dim">'
            '    <Dim name="/y" />'
            '  </Float64>'
            '  <Float64 name="variable_without_x_y_dims">'
            '    <Dim name="/lon" />'
            '    <Dim name="/lat" />'
            '  </Float64>'
            '</Dataset>'
        )

        dmr_path = path_join(self.temp_dir, 'x_y_dimensions.dmr.xml')

        with open(dmr_path, 'w', encoding='utf-8') as file_handler:
            file_handler.write(sample_dmr)

        varinfo = VarInfoFromDmr(dmr_path)
        expected_x = '/x'
        expected_y = '/y'

        with self.subTest('A variable has both x and y dimensions'):
            actual_x, actual_y = get_projected_x_y_variables(
                varinfo, '/variable_with_x_y_dims'
            )
            self.assertEqual(actual_x, expected_x)
            self.assertEqual(actual_y, expected_y)

        with self.subTest('Variable lacks projection_x_coordinate dimension'):
            actual_x, actual_y = get_projected_x_y_variables(
                varinfo, '/variable_with_y_dim'
            )
            self.assertIsNone(actual_x)
            self.assertEqual(actual_y, expected_y)

        with self.subTest('Variable lacks projection_y_coordinate dimension'):
            actual_x, actual_y = get_projected_x_y_variables(
                varinfo, '/variable_with_x_dim'
            )
            self.assertEqual(actual_x, expected_x)
            self.assertIsNone(actual_y)

        with self.subTest('Variable lacks x and y dimensions'):
            actual_x, actual_y = get_projected_x_y_variables(
                varinfo, '/variable_without_x_y_dims'
            )
            self.assertIsNone(actual_x)
            self.assertIsNone(actual_y)

    def test_is_projection_x_dimension(self):
        """Ensure that a dimension variable is correctly identified as being
        an x-dimension if it has the expected `standard_name`. This
        function must also handle absent dimensions, for cases such as the
        `nv`, `latv` or `lonv` dimensions that do not have corresponding
        variables in a granule.

        """
        sample_dmr = (
            '<Dataset xmlns="namespace_string">'
            '  <Dimension name="x" size="500" />'
            '  <Float64 name="x">'
            '    <Dim name="/x" />'
            '    <Attribute name="standard_name" type="String">'
            '      <Value>projection_x_coordinate</Value>'
            '    </Attribute>'
            '  </Float64>'
            '  <Float64 name="y">'
            '    <Dim name="/y" />'
            '    <Attribute name="standard_name" type="String">'
            '      <Value>projection_y_coordinate</Value>'
            '    </Attribute>'
            '  </Float64>'
            '</Dataset>'
        )

        dmr_path = path_join(self.temp_dir, 'x_y_dimensions.dmr.xml')

        with open(dmr_path, 'w', encoding='utf-8') as file_handler:
            file_handler.write(sample_dmr)

        varinfo = VarInfoFromDmr(dmr_path)

        with self.subTest('x-dimension returns True'):
            self.assertTrue(is_projection_x_dimension(varinfo, '/x'))

        with self.subTest('Non x-dimension returns False'):
            self.assertFalse(is_projection_x_dimension(varinfo, '/y'))

        with self.subTest('Non-existent dimension returns False'):
            self.assertFalse(is_projection_x_dimension(varinfo, '/missing'))

    def test_is_projection_y_variable(self):
        """Ensure that a dimension variable is correctly identified as being
        an y-dimension if it has the expected `standard_name`. This
        function must also handle absent dimensions, for cases such as the
        `nv`, `latv` or `lonv` dimensions that do not have corresponding
        variables in a granule.

        """
        sample_dmr = (
            '<Dataset xmlns="namespace_string">'
            '  <Dimension name="x" size="500" />'
            '  <Float64 name="x">'
            '    <Dim name="/x" />'
            '    <Attribute name="standard_name" type="String">'
            '      <Value>projection_x_coordinate</Value>'
            '    </Attribute>'
            '  </Float64>'
            '  <Float64 name="y">'
            '    <Dim name="/y" />'
            '    <Attribute name="standard_name" type="String">'
            '      <Value>projection_y_coordinate</Value>'
            '    </Attribute>'
            '  </Float64>'
            '</Dataset>'
        )

        dmr_path = path_join(self.temp_dir, 'x_y_dimensions.dmr.xml')

        with open(dmr_path, 'w', encoding='utf-8') as file_handler:
            file_handler.write(sample_dmr)

        varinfo = VarInfoFromDmr(dmr_path)

        with self.subTest('y-dimension returns True'):
            self.assertTrue(is_projection_y_dimension(varinfo, '/y'))

        with self.subTest('Non y-dimension returns False'):
            self.assertFalse(is_projection_y_dimension(varinfo, '/x'))

        with self.subTest('Non-existent dimension returns False'):
            self.assertFalse(is_projection_y_dimension(varinfo, '/missing'))

    def test_get_grid_lat_lons(self):
        """Ensure that a grid of projected values is correctly converted to
        longitude and latitude values. The inputs include 1-D arrays for
        the x and y dimensions, whilst the output are 2-D grids of latitude
        and longitude that correspond to all grid points defined by the
        combinations of x and y coordinates.

        """
        x_values = np.array([1513760.59366167, 1048141.65434399])
        y_values = np.array([-705878.15743769, -381492.36347575])
        crs = CRS.from_epsg(6931)

        actual_lats, actual_lons = get_grid_lat_lons(x_values, y_values, crs)
        expected_lats = np.array([[75.0, 78.6663628], [75.9858088, 80.0]])
        expected_lons = np.array([[65.0, 56.0414351], [75.8550777, 70.0]])

        np.testing.assert_almost_equal(actual_lats, expected_lats)
        np.testing.assert_almost_equal(actual_lons, expected_lons)

    def test_get_geographic_resolution(self):
        """Ensure the calculated resolution is the minimum Euclidean distance
        between diagonally adjacent pixels.

        The example coordinates below have the shortest diagonal difference
        between (10, 10) and (15, 15), resulting in a resolution of
        (5^2 + 5^2)^0.5 = 50^0.5 ~= 7.07.

        """
        latitudes = np.array([[10, 10, 10], [15, 15, 15], [25, 25, 25]])
        longitudes = np.array([[10, 15, 25], [10, 15, 25], [10, 15, 25]])
        expected_resolution = 7.071
        self.assertAlmostEqual(
            get_geographic_resolution(longitudes, latitudes),
            expected_resolution,
            places=3,
        )

    @patch('hoss.projection_utilities.get_bbox_polygon')
    @patch('hoss.projection_utilities.get_resolved_feature')
    @patch('hoss.projection_utilities.get_resolved_features')
    def test_get_resolved_geojson(
        self,
        mock_get_resolved_features,
        mock_get_resolved_feature,
        mock_get_bbox_polygon,
    ):
        """Ensure that a GeoJSON shape or bounding box is correctly resolved
        using the correct functionality (bounding box versus shape file).

        """
        resolution = 0.1
        shape_file = f'tests/geojson_examples/{self.polygon_file_name}'
        bounding_box = BBox(0, 10, 20, 30)
        bounding_box_polygon = Polygon([(0, 10), (20, 10), (20, 30), (0, 30), (0, 10)])
        resolved_feature = [(0, 10), (20, 10), (20, 30), (0, 30)]
        resolved_features = [
            (-114.05, 42.0),
            (-114.05, 37.0),
            (-109.04, 37.0),
            (-109.04, 41.0),
            (-111.05, 41.0),
        ]

        mock_get_resolved_features.return_value = resolved_features
        mock_get_resolved_feature.return_value = resolved_feature
        mock_get_bbox_polygon.return_value = bounding_box_polygon

        with self.subTest('Shape file is specified and used'):
            self.assertListEqual(
                get_resolved_geojson(resolution, shape_file=shape_file),
                resolved_features,
            )
            mock_get_resolved_features.assert_called_once_with(
                self.polygon_geojson, resolution
            )
            mock_get_resolved_feature.assert_not_called()
            mock_get_bbox_polygon.assert_not_called()

        mock_get_resolved_features.reset_mock()

        with self.subTest('Bounding box is specified and used'):
            self.assertListEqual(
                get_resolved_geojson(resolution, bounding_box=bounding_box),
                resolved_feature,
            )
            mock_get_resolved_features.assert_not_called()
            mock_get_resolved_feature.assert_called_once_with(
                bounding_box_polygon, resolution
            )
            mock_get_bbox_polygon.assert_called_once_with(bounding_box)

        mock_get_resolved_feature.reset_mock()
        mock_get_bbox_polygon.reset_mock()

        with self.subTest('Bounding box is used when both are specified'):
            self.assertListEqual(
                get_resolved_geojson(
                    resolution, shape_file=shape_file, bounding_box=bounding_box
                ),
                resolved_feature,
            )
            mock_get_resolved_feature.assert_called_once_with(
                bounding_box_polygon, resolution
            )
            mock_get_bbox_polygon.assert_called_once_with(bounding_box)
            mock_get_resolved_features.assert_not_called()

        mock_get_resolved_feature.reset_mock()

        with self.subTest('Neither shape file nor bbox, raises exception'):
            with self.assertRaises(MissingSpatialSubsetInformation):
                get_resolved_geojson(resolution, None, None)
                mock_get_resolved_features.assert_not_called()
                mock_get_bbox_polygon.assert_not_called()
                mock_get_resolved_feature.assert_not_called()

    def test_get_bbox_polygon(self):
        """Ensure a polygon is constructed from the input bounding box. It
        should only have an exterior set of points, and those should only
        be combinations of the West, South, East and North coordinates of
        the input bounding box.

        """
        bounding_box = BBox(0, 10, 20, 30)
        expected_bounding_box_polygon = Polygon(
            [(0, 10), (20, 10), (20, 30), (0, 30), (0, 10)]
        )
        bounding_box_result = get_bbox_polygon(bounding_box)
        self.assertEqual(bounding_box_result, expected_bounding_box_polygon)
        self.assertListEqual(list(bounding_box_result.interiors), [])

    @patch('hoss.projection_utilities.get_resolved_feature')
    def test_get_resolved_features(self, mock_get_resolved_feature):
        """Ensure that the parsed GeoJSON content can be correctly sent to
        `get_resolved_feature`, depending on if the content is a GeoJSON
        Geometry, Feature or FeatureCollection. If the object does not
        conform to the expected GeoJSON schema, and exception will be
        raised.

        """
        resolution = 2.0
        resolved_linestring = [
            (-75.696, 38.471),
            (-75.795, 39.716),
            (-77.370, 39.719),
            (-78.944, 39.721),
            (-80.519, 39.724),
        ]

        with self.subTest('A Geometry input is passed directly through'):
            mock_get_resolved_feature.return_value = resolved_linestring
            self.assertListEqual(
                get_resolved_features(
                    self.linestring_geojson['features'][0]['geometry'], resolution
                ),
                resolved_linestring,
            )
            mock_get_resolved_feature.assert_called_once_with(
                self.linestring, resolution
            )

        mock_get_resolved_feature.reset_mock()

        with self.subTest('A Feature input uses its Geometry attribute'):
            mock_get_resolved_feature.return_value = resolved_linestring
            self.assertListEqual(
                get_resolved_features(
                    self.linestring_geojson['features'][0], resolution
                ),
                resolved_linestring,
            )
            mock_get_resolved_feature.assert_called_once_with(
                self.linestring, resolution
            )

        mock_get_resolved_feature.reset_mock()

        # GeoJSON with a list of multiple features:
        multi_feature_geojson = self.read_geojson('multi_feature.geo.json')
        first_shape = shape(multi_feature_geojson['features'][0]['geometry'])
        second_shape = shape(multi_feature_geojson['features'][1]['geometry'])
        multi_feature_side_effect = [
            [(-75.565, 39.662)],
            [
                (-75.696, 38.471),
                (-75.795, 39.716),
                (-77.370, 39.718),
                (-78.944, 39.721),
                (-80.519, 39.724),
            ],
        ]
        resolved_multi_feature = [
            (-75.565, 39.662),
            (-75.696, 38.471),
            (-75.795, 39.716),
            (-77.370, 39.718),
            (-78.944, 39.721),
            (-80.519, 39.724),
        ]

        with self.subTest('A FeatureCollection uses the Geometry of each Feature'):
            mock_get_resolved_feature.side_effect = multi_feature_side_effect
            self.assertListEqual(
                get_resolved_features(multi_feature_geojson, resolution),
                resolved_multi_feature,
            )
            self.assertEqual(mock_get_resolved_feature.call_count, 2)
            mock_get_resolved_feature.assert_has_calls(
                [call(first_shape, resolution), call(second_shape, resolution)]
            )

        mock_get_resolved_feature.reset_mock()

        with self.subTest('Unexpected schema'):
            with self.assertRaises(InvalidInputGeoJSON):
                get_resolved_features({'random': 'json'}, resolution)

    @patch('hoss.projection_utilities.get_resolved_geometry')
    def test_get_resolved_feature(self, mock_get_resolved_geometry):
        """Ensure that GeoJSON features with various geometry types are
        correctly handled to produce a list of points at the specified
        resolution.

        Single geometry features (Point, Line, Polygon) should be handled
        with a single call to `get_resolved_feature`.

        Multi geometry features (MultiPoint, Line, Polygon,
        GeometryCollection) should recursively call this function and
        flatten the resulting list of lists of coordinates.

        Any other geometry type will not be recognised and will raise an
        exception.

        Mock return values for `get_resolved_geometry` are rounded to 2 or
        3 decimal places as appropriate, but are otherwise accurate.

        """
        resolution = 2.0
        resolved_polygon = [
            (-114.05, 42.0),
            (-114.05, 40.33),
            (-114.05, 38.67),
            (-114.05, 37.0),
            (-112.38, 37.0),
            (-110.71, 37.0),
            (-109.04, 37.0),
            (-109.04, 39.0),
            (-109.04, 41.0),
            (-110.045, 41.0),
            (-111.05, 41.0),
            (-111.05, 42.0),
            (-112.55, 42.0),
        ]
        resolved_linestring = [
            (-75.696, 38.471),
            (-75.795, 39.716),
            (-77.370, 39.719),
            (-78.944, 39.721),
            (-80.519, 39.724),
        ]

        mlinestring_side_effect = [
            [(-3.194, 55.949), (-3.181, 55.951), (-3.174, 55.953)],
            [(-0.14, 51.502), (-0.128, 51.507)],
        ]
        resolved_mlinestring = [
            (-3.194, 55.949),
            (-3.181, 55.951),
            (-3.174, 55.953),
            (-0.14, 51.502),
            (-0.128, 51.507),
        ]

        resolved_multi_point = [(-0.076, 51.508), (-0.142, 51.501)]

        mpolygon_side_effect = [
            [
                (-109.05, 41.0),
                (-109.05, 39.0),
                (-109.05, 37),
                (-105.55, 37.0),
                (-103.8, 37.0),
                (-102.05, 37.0),
                (-102.05, 39.0),
                (-102.05, 41.0),
                (-103.8, 41.0),
                (-105.55, 41.0),
                (-107.3, 41.0),
            ]
        ]
        resolved_mpolygon = mpolygon_side_effect[0]

        geom_coll_side_effect = [
            [
                (-75.696, 38.471),
                (-75.795, 39.716),
                (-77.370, 39.718),
                (-78.944, 39.721),
                (-80.519, 39.724),
            ]
        ]
        resolved_geom_collection = [
            (-75.565, 39.662),
            (-75.696, 38.471),
            (-75.795, 39.716),
            (-77.370, 39.718),
            (-78.944, 39.721),
            (-80.519, 39.724),
        ]

        with self.subTest('Polygon'):
            mock_get_resolved_geometry.return_value = resolved_polygon
            self.assertListEqual(
                get_resolved_feature(self.polygon, resolution), resolved_polygon
            )
            mock_get_resolved_geometry.assert_called_once_with(
                list(self.polygon.exterior.coords), resolution
            )

        mock_get_resolved_geometry.reset_mock()

        with self.subTest('LineString'):
            mock_get_resolved_geometry.return_value = resolved_linestring
            self.assertListEqual(
                get_resolved_feature(self.linestring, resolution), resolved_linestring
            )
            mock_get_resolved_geometry.assert_called_once_with(
                list(self.linestring.coords), resolution, is_closed=False
            )

        mock_get_resolved_geometry.reset_mock()

        with self.subTest('Point'):
            self.assertListEqual(
                get_resolved_feature(self.point, resolution),
                [(self.point.x, self.point.y)],
            )
            mock_get_resolved_geometry.assert_not_called()

        with self.subTest('MultiPolygon'):
            mock_get_resolved_geometry.side_effect = mpolygon_side_effect
            self.assertListEqual(
                get_resolved_feature(self.multi_polygon, resolution), resolved_mpolygon
            )
            mock_get_resolved_geometry.assert_called_once_with(
                list(self.multi_polygon.geoms[0].exterior.coords),
                resolution,
            )

        mock_get_resolved_geometry.reset_mock()

        with self.subTest('MultiLineString'):
            mock_get_resolved_geometry.side_effect = mlinestring_side_effect
            self.assertListEqual(
                get_resolved_feature(self.multi_linestring, resolution),
                resolved_mlinestring,
            )
            self.assertEqual(mock_get_resolved_geometry.call_count, 2)
            mock_get_resolved_geometry.assert_has_calls(
                [
                    call(
                        list(self.multi_linestring.geoms[0].coords),
                        resolution,
                        is_closed=False,
                    ),
                    call(
                        list(self.multi_linestring.geoms[1].coords),
                        resolution,
                        is_closed=False,
                    ),
                ]
            )

        mock_get_resolved_geometry.reset_mock()

        with self.subTest('MultiPoint'):
            self.assertListEqual(
                get_resolved_feature(self.multi_point, resolution), resolved_multi_point
            )
            mock_get_resolved_geometry.assert_not_called()

        with self.subTest('GeometryCollection'):
            # Contains a Point and a LineString, the point will not need to
            # call `get_resolved_geometry`.
            mock_get_resolved_geometry.side_effect = geom_coll_side_effect
            self.assertListEqual(
                get_resolved_feature(self.geometry_coll, resolution),
                resolved_geom_collection,
            )
            mock_get_resolved_geometry.assert_called_once_with(
                list(self.geometry_coll.geoms[1].coords), resolution, is_closed=False
            )

        mock_get_resolved_geometry.reset_mock()

        with self.subTest('Unknown feature'):
            with self.assertRaises(InvalidInputGeoJSON):
                get_resolved_feature('not_geojson_shape', resolution)

    def test_get_resolved_geometry(self):
        """Ensure that a set of input points are updated to the specified
        resolution. Specific test cases include whether the input forms a
        closed loop or not.

        """
        input_geometry = [(1.0, 1.0), (1.0, 1.5), (2.0, 1.5), (2.0, 1.0), (1.0, 1.0)]
        resolution = 0.5
        output_open_geometry = [
            (1.0, 1.0),
            (1.0, 1.5),
            (1.5, 1.5),
            (2.0, 1.5),
            (2.0, 1.0),
            (1.5, 1.0),
            (1.0, 1.0),
        ]
        output_closed_geometry = output_open_geometry[:-1]

        test_args = [
            ['Open geometry includes the final point.', False, output_open_geometry],
            ['Closed geometry excludes final point.', True, output_closed_geometry],
        ]

        for description, is_closed, expected_geometry in test_args:
            with self.subTest(description):
                self.assertListEqual(
                    get_resolved_geometry(
                        input_geometry, resolution, is_closed=is_closed
                    ),
                    expected_geometry,
                )

    def test_get_resolved_line(self):
        """Ensure that a line, defined by its two end-points, will be
        converted so that there are evenly spaced points separated by,
        at most, the resolution supplied to the function.

        Note, in the first test, the distance between each point is 2.83,
        resulting from the smallest number of points possible being placed
        on the line at a distance of no greater than the requested
        resolution (3).

        """
        test_args = [
            [
                'Line needs additional points',
                (0, 0),
                (10, 10),
                3,
                [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8), (10, 10)],
            ],
            ['Resolution bigger than line', (0, 0), (1, 1), 2, [(0, 0), (1, 1)]],
            [
                'Line flat in one dimension',
                (0, 0),
                (0, 10),
                5,
                [(0, 0), (0, 5), (0, 10)],
            ],
        ]

        for description, point_one, point_two, resolution, expected_output in test_args:
            with self.subTest(description):
                self.assertListEqual(
                    get_resolved_line(point_one, point_two, resolution), expected_output
                )

    def test_get_x_y_extents_from_geographic_points(self):
        """Ensure that a list of coordinates is transformed to a specified
        projection, and that the expected extents in the projected x and y
        dimensions are returned.

        """
        points = [(-180, 75), (-90, 75), (0, 75), (90, 75)]
        crs = CRS.from_epsg(6931)
        expected_x_y_extents = {
            'x_min': -1670250.0136418417,
            'x_max': 1670250.0136418417,
            'y_min': -1670250.0136418417,
            'y_max': 1670250.0136418417,
        }

        self.assertDictEqual(
            get_x_y_extents_from_geographic_points(points, crs), expected_x_y_extents
        )
