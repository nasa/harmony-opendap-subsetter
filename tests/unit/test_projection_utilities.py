"""This module tests the functions included in the
`hoss/projection_utilities.py` module. Those functions focus on using the
spatial constraint information from an input Harmony message with
collections that have projected grids.

"""

import json
import math
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
    InvalidGranuleDimensions,
    InvalidInputGeoJSON,
    InvalidRequestedRange,
    MissingGridMappingMetadata,
    MissingGridMappingVariable,
    MissingSpatialSubsetInformation,
)
from hoss.projection_utilities import (
    get_bbox_polygon,
    get_densified_perimeter,
    get_filtered_points,
    get_geographic_resolution,
    get_grid_lat_lons,
    get_grid_mapping_attributes,
    get_master_geotransform,
    get_projected_x_y_extents,
    get_projected_x_y_variables,
    get_resolved_feature,
    get_resolved_features,
    get_resolved_geometry,
    get_resolved_line,
    get_variable_crs,
    get_x_y_extents_from_geographic_perimeter,
    is_projection_x_dimension,
    is_projection_y_dimension,
    perimeter_surrounds_grid,
    remove_non_finite_projected_values,
    remove_points_outside_grid_extents,
)
from tests import utilities
from tests.utilities import assert_float_dict_almost_equal


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

    @patch('hoss.projection_utilities.get_grid_mapping_attributes')
    def test_get_variable_crs(self, mock_get_grid_mapping_attributes):
        """Ensure a `pyproj.CRS` object can be instantiated from the given
        `grid_mapping_attributes`

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
            '</Dataset>'
        )

        dmr_path = path_join(self.temp_dir, 'grid_mapping.dmr.xml')

        with open(dmr_path, 'w', encoding='utf-8') as file_handler:
            file_handler.write(sample_dmr)

        varinfo = VarInfoFromDmr(dmr_path)
        grid_mapping_attributes = {
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

        mock_get_grid_mapping_attributes.return_value = grid_mapping_attributes

        expected_crs = CRS.from_cf(grid_mapping_attributes)

        with self.subTest('Variable with "grid_mapping" gets expected CRS'):
            actual_crs = get_variable_crs('/variable_with_grid_mapping', varinfo)
            self.assertEqual(actual_crs, expected_crs)
            self.assertIsInstance(actual_crs, CRS)

    def test_get_grid_mapping_attributes(self):
        """Ensure that the grid mapping attributes can be retrieved via the reference
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

        expected_grid_mapping_attributes = {
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

        with self.subTest(
            'Variable with "grid_mapping" gets expected grid mapping attributes'
        ):
            actual_grid_mapping_attributes = get_grid_mapping_attributes(
                '/variable_with_grid_mapping', varinfo
            )
            self.assertEqual(
                actual_grid_mapping_attributes, expected_grid_mapping_attributes
            )

        with self.subTest('Variable has no "grid_mapping" attribute'):
            with self.assertRaises(MissingGridMappingMetadata) as context:
                get_grid_mapping_attributes('/variable_without_grid_mapping', varinfo)

            self.assertEqual(
                context.exception.message,
                'Projected variable "/variable_without_grid_mapping"'
                ' does not have an associated "grid_mapping" '
                'metadata attribute.',
            )

        with self.subTest('"grid_mapping" points to non-existent variable'):
            with self.assertRaises(MissingGridMappingVariable) as context:
                get_grid_mapping_attributes('/variable_with_bad_grid_mapping', varinfo)

            self.assertEqual(
                context.exception.message,
                'Grid mapping variable "/non_existent_crs" '
                'referred to by variable '
                '"/variable_with_bad_grid_mapping" is not '
                'present in granule .dmr file.',
            )

        with self.subTest(
            'attributes for missing grid_mapping retrieved from earthdata-varinfo configuration file'
        ):
            smap_varinfo = VarInfoFromDmr(
                'tests/data/SC_SPL3SMP_008.dmr',
                'SPL3SMP',
                'hoss/hoss_config.json',
            )
            expected_grid_mapping_attributes = {
                'false_easting': 0.0,
                'false_northing': 0.0,
                'longitude_of_central_meridian': 0.0,
                'standard_parallel': 30.0,
                'grid_mapping_name': 'lambert_cylindrical_equal_area',
            }

            actual_grid_mapping_attributes = get_grid_mapping_attributes(
                '/Soil_Moisture_Retrieval_Data_AM/surface_flag', smap_varinfo
            )
            # self.assertEqual(
            #     actual_grid_mapping_attributes, expected_grid_mapping_attributes
            # )

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

            assert_float_dict_almost_equal(
                get_projected_x_y_extents(
                    x_values, y_values, crs, bounding_box=bounding_box
                ),
                expected_output,
            )

        with self.subTest('Shape file input'):
            assert_float_dict_almost_equal(
                get_projected_x_y_extents(
                    x_values, y_values, crs, shape_file=polygon_path
                ),
                expected_output,
            )

    def test_get_projected_x_y_extents_whole_earth(self):
        """Ensure that the expected values for the x and y dimension extents
        are recovered for a polar projected grid and when a whole earth
        bounding box or shape is requested.

        """
        whole_earth_bbox = BBox(-180.0, -90.0, 180.0, 90.0)

        polygon_path = 'tests/geojson_examples/polygon_whole_earth.geo.json'

        x_values = np.linspace(-8982000, 8982000, 500)
        y_values = np.linspace(8982000, -8982000, 500)

        crs = CRS.from_cf(
            {
                'false_easting': 0.0,
                'false_northing': 0.0,
                'longitude_of_central_meridian': 0.0,
                'latitude_of_projection_origin': 90.0,
                'grid_mapping_name': 'lambert_azimuthal_equal_area',
            }
        )
        expected_output = {
            'x_min': -8982000,
            'x_max': 8982000,
            'y_min': -8982000,
            'y_max': 8982000,
        }
        with self.subTest('Whole Earth LAEA - Bounding box input'):
            assert_float_dict_almost_equal(
                get_projected_x_y_extents(
                    x_values, y_values, crs, bounding_box=whole_earth_bbox
                ),
                expected_output,
            )

        with self.subTest('Whole Earth LAEA - Shape file input'):
            assert_float_dict_almost_equal(
                get_projected_x_y_extents(
                    x_values, y_values, crs, shape_file=polygon_path
                ),
                expected_output,
            )

    def test_get_projected_x_y_extents_edge_case(self):
        """Ensure that the expected values for the x and y dimension extents
        are recovered for a polar projected grid and when a bounding box
        for any edge cases are requested.

        """
        bbox = BBox(-89, -79, -40, -59)

        x_values = np.linspace(-9000000, 9000000, 2000)
        y_values = np.linspace(9000000, -9000000, 2000)

        crs = CRS.from_cf(
            {
                'false_easting': 0.0,
                'false_northing': 0.0,
                'longitude_of_central_meridian': 0.0,
                'latitude_of_projection_origin': 90.0,
                'grid_mapping_name': 'lambert_azimuthal_equal_area',
            }
        )

        expected_output = {
            'x_min': -8993061.78423412,
            'x_max': -8350580.505440015,
            'y_min': -8997181.591145469,
            'y_max': -8354987.361637551,
        }
        with self.subTest(
            'LAEA - Bounding box which is close to the edge of granule extent'
        ):
            assert_float_dict_almost_equal(
                get_projected_x_y_extents(x_values, y_values, crs, bounding_box=bbox),
                expected_output,
            )

        bbox = BBox(-90, -87, -75, -85)
        with self.subTest('LAEA - Bounding box which is outside the granule extent'):
            with self.assertRaises(InvalidRequestedRange):
                get_projected_x_y_extents(x_values, y_values, crs, bounding_box=bbox)

        with self.subTest('When the granule has invalid dimensions'):
            with self.assertRaises(InvalidGranuleDimensions):
                x_values1 = np.linspace(-9200000, 9200000, 500)
                y_values1 = np.linspace(9200000, -9200000, 500)
                get_projected_x_y_extents(x_values1, y_values1, crs, bounding_box=bbox)

    def test_get_filtered_points(self):
        """Ensure that the coordinates returned are clipped to the granule extent or
        the bbox extent whichever is the smaller of the two.

        """
        granule_extent = BBox(-120.0, -80.0, 120.0, 80.0)
        granule_extent_points = [
            (-120.0, -80.0),
            (120.0, -80.0),
            (120.0, 80.0),
            (-120.0, 80.0),
            (-120.0, -80.0),
        ]
        bounding_points_contains_granule = [
            (-170.0, -85.0),
            (170.0, -85.0),
            (170.0, 85.0),
            (-170.0, 85.0),
            (-170.0, -85.0),
        ]

        bounding_points_within_granule = [
            (-70.0, -70.0),
            (70.0, -70.0),
            (70.0, 70.0),
            (-70.0, 70.0),
            (-70.0, -70.0),
        ]

        bounding_points_not_overlapping_granule = [
            (-179.98, -87.0),
            (179.89, -87.0),
            (179.89, 88.0),
            (-179.98, 88.0),
            (-179.98, -87.0),
        ]

        some_points_overlapping_granule = [
            (-100.0, -60.0),
            (100, -60.0),
            (179.91, 88.0),
            (-100, 80.0),
            (-100.0, -60.0),
        ]

        with self.subTest('Bounding box completely contains granule extent'):
            self.assertListEqual(
                get_filtered_points(bounding_points_contains_granule, granule_extent),
                granule_extent_points,
            )

        with self.subTest('Bounding box entirely within the granule extent'):
            self.assertListEqual(
                get_filtered_points(bounding_points_within_granule, granule_extent),
                bounding_points_within_granule,
            )

        with self.subTest('Bounding box and granule do not overlap'):
            self.assertListEqual(
                get_filtered_points(
                    bounding_points_not_overlapping_granule, granule_extent
                ),
                granule_extent_points,
            )

        with self.subTest('Bounding box input with some points overlapping granule'):
            self.assertListEqual(
                get_filtered_points(some_points_overlapping_granule, granule_extent),
                [
                    (-100.0, -60.0),
                    (100.0, -60.0),
                    (120.0, 80.0),
                    (-100.0, 80.0),
                    (-100.0, -60.0),
                ],
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
    def test_get_densified_perimeter(
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
                get_densified_perimeter(resolution, shape_file=shape_file),
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
                get_densified_perimeter(resolution, bounding_box=bounding_box),
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
                get_densified_perimeter(
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
                get_densified_perimeter(resolution, None, None)
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

    def test_get_x_y_extents_from_geographic_perimeter(self):
        """Ensure that a list of coordinates is transformed to a specified
        projection, and that the expected extents in the projected x and y
        dimensions are returned.

        """

        granule_extent = {
            "x_min": -9000000,
            "x_max": 9000000,
            "y_min": -9000000,
            "y_max": 9000000,
        }
        points = [(-180, 75), (-90, 75), (0, 75), (90, 75)]
        crs = CRS.from_epsg(6931)
        expected_x_y_extents = {
            'x_min': -1670250.0136418417,
            'x_max': 1670250.0136418417,
            'y_min': -1670250.0136418417,
            'y_max': 1670250.0136418417,
        }

        assert_float_dict_almost_equal(
            get_x_y_extents_from_geographic_perimeter(points, crs, granule_extent),
            expected_x_y_extents,
        )

    def test_get_x_y_extents_from_geographic_perimeter_full_earth_laea(self):
        """Ensure that a list of coordinates is transformed to the specified
        laea projection, and valid values in the projected x and y
        dimensions are returned even for edge cases like whole earth.

        """
        granule_extent_projected_meters = {
            "x_min": -9000000,
            "x_max": 9000000,
            "y_min": -9000000,
            "y_max": 9000000,
        }
        crs = CRS.from_cf(
            {
                'false_easting': 0.0,
                'false_northing': 0.0,
                'longitude_of_central_meridian': 0.0,
                'latitude_of_projection_origin': 90.0,
                'grid_mapping_name': 'lambert_azimuthal_equal_area',
            }
        )

        points1 = [(-180, -90), (-180, 90), (180, 90), (180, -90)]
        x_y_extents = get_x_y_extents_from_geographic_perimeter(
            points1, crs, granule_extent_projected_meters
        )

        self.assertTrue(all(not math.isinf(value) for value in x_y_extents.values()))

    def test_remove_non_finite_projected_values(self):
        """Ensure that only valid values in the x,y list are returned and any
        NaN or inf values are removed.
        """
        points_x = [
            float('-inf'),
            -900.3,
            -800.2,
            -700.1,
            float('nan'),
            500.0,
            600.9,
            700.0,
            800.0,
        ]
        points_y = [
            89.0,
            float('nan'),
            69.1,
            40.5,
            -10.6,
            50.9,
            70.2,
            80.4,
            float('inf'),
        ]
        expected_x_values = np.array([-800.2, -700.1, 500.0, 600.9, 700.0])
        expected_y_values = np.array([69.1, 40.5, 50.9, 70.2, 80.4])
        valid_x, valid_y = remove_non_finite_projected_values(points_x, points_y)
        assert np.array_equal(valid_x, expected_x_values)
        assert np.array_equal(valid_y, expected_y_values)

    def test_check_perimeter_exceeds_grid_extents(self):
        """Ensure that True value is returned when the input perimeter array exceeds grid
        extents and a False value is returned when the perimeter is within the grid extents

        """
        granule_extent = {
            "x_min": -1000.0,
            "x_max": 1000.0,
            "y_min": -1000.0,
            "y_max": 1000.0,
        }
        with self.subTest("Perimeter exceeds grid extents in all axes"):
            self.assertTrue(
                perimeter_surrounds_grid(
                    np.array(
                        [-2000.1, -1000.1, -900, -500, 200.3, 700.1, 800.1, 1200.5]
                    ),
                    np.array(
                        [1100.1, 400.9, 200, -100.8, -300.3, -500.1, -600.1, -1000.5]
                    ),
                    granule_extent,
                )
            )
        with self.subTest("Perimeter does not surround grid"):
            self.assertFalse(
                perimeter_surrounds_grid(
                    np.array(
                        [-2000.1, -1000.1, -900, -500, 200.3, 700.1, 800.1, 900.0]
                    ),
                    np.array(
                        [400.1, 300.9, 200, -100.8, -300.3, -500.1, -600.1, -700.5]
                    ),
                    granule_extent,
                )
            )

    def test_remove_points_outside_grid_extents(self):
        """Ensure that any point outside the grid extents and removed and the grid returned
        only has values within the grid extent.

        """
        points_x = np.array(
            [
                -1100.1,  # 1 should exclude this < x_min
                -1000.0123456783,  # 2 should include this - edge case ~ xmin
                1000.0123456798,  # 3 should exclude this - edge case > xmax,
                1000.0123456781,  # 4 should include this - edge case ~ xmax
                -700.5,  # 5
                -700.6,  # 6
                -700.7,  # 7
                -800.8,  # 8
                -900.9,  # 9
                -950.1,  # 10
                -1000.0123456797,  # 11 should exclude this - edge case < xmin
                1100.1,  # 12 should exclude this > x_max
            ]
        )
        points_y = np.array(
            [
                600.1,  # 1
                700.2,  # 2
                800.3,  # 3
                900.4,  # 4
                1000.0123456781,  # 5 should include this < y_max edge case
                1100.6,  # 6 exclude this > y_max
                1000.0123456799,  # 7 should exclude this > y_max edge case
                -1000.0123456785,  # 8 should include this ~ y_min
                -1000.0123456796,  # 9 should exclude this < y_min edge case
                -1002.1,  # 10 exclude < y_min
                700.0,  # 11
                600.1,  # 12
            ]
        )
        expected_x = np.array(
            [
                -1000.012345678,  # 2
                1000.0123456781,  # 4
                -700.5,  # 5
                -800.8,  # 8
            ]
        )
        expected_y = np.array(
            [700.2, 900.4, 1000.012345678, -1000.012345678]  # 2  # 4  # 6  # 8
        )

        granule_extent = {
            "x_min": -1000.012345678,
            "x_max": 1000.012345678,
            "y_min": -1000.012345678,
            "y_max": 1000.012345678,
        }

        with self.subTest("Perimeter contains some valid points within grid extent"):
            x, y = remove_points_outside_grid_extents(
                points_x, points_y, granule_extent
            )

            assert np.allclose(expected_x, x, atol=1e-9, rtol=0)
            assert np.allclose(expected_y, y, atol=1e-9, rtol=0)

        with self.subTest("Perimeter has no valid points within the grid extent"):
            with self.assertRaises(InvalidRequestedRange):
                remove_points_outside_grid_extents(
                    np.array([-2000.1, -1000.1, -900, -500, 200.3, 700.1, 1200.5]),
                    np.array([600, 800.9, 1000.2, 1100.8, 1100.1, 1000.1, 900.5]),
                    granule_extent,
                )

    @patch('hoss.projection_utilities.get_grid_mapping_attributes')
    def test_get_master_geotransform(self, mock_get_grid_mapping_attributes):
        """Ensure that the `master_geotransform` attribute is returned. If it doesn't
        exist the return value should be `None`.
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
            '</Dataset>'
        )

        dmr_path = path_join(self.temp_dir, 'grid_mapping.dmr.xml')

        with open(dmr_path, 'w', encoding='utf-8') as file_handler:
            file_handler.write(sample_dmr)

        varinfo = VarInfoFromDmr(dmr_path)

        with self.subTest('grid mapping attributes contain master geotransform'):
            mock_get_grid_mapping_attributes.return_value = {
                'master_geotransform': [-9000000, 3000, 0, 9000000, 0, -3000]
            }
            result = get_master_geotransform("test_variable", varinfo)
            self.assertEqual(result, [-9000000, 3000, 0, 9000000, 0, -3000])

        with self.subTest('grid mapping attributes do not contain master geotransform'):
            mock_get_grid_mapping_attributes.return_value = {
                'fake_attribute': [-9000000, 3000, 0, 9000000, 0, -3000]
            }
            result = get_master_geotransform("test_variable", varinfo)
            self.assertIsNone(result)
