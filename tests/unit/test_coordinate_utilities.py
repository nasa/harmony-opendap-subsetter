from logging import getLogger
from os.path import exists
from unittest import TestCase
from unittest.mock import ANY, patch

import numpy as np
from harmony.util import config
from netCDF4 import Dataset
from numpy.testing import assert_array_equal
from pyproj import CRS
from varinfo import VarInfoFromDmr

from hoss.coordinate_utilities import (
    any_absent_dimension_variables,
    get_1d_dim_array_data_from_dimvalues,
    get_coordinate_array,
    get_coordinate_variables,
    get_dimension_array_from_geo_points,
    get_max_x_spread_pts,
    get_projected_dimension_names,
    get_projected_dimension_names_from_coordinate_variables,
    get_row_col_sizes_from_coordinate_datasets,
    get_valid_indices,
    get_valid_row_col_pairs,
    get_variables_with_anonymous_dims,
    get_x_y_values_from_geographic_points,
)
from hoss.exceptions import (
    IncompatibleCoordinateVariables,
    InvalidCoordinateData,
    InvalidCoordinateVariable,
    MissingCoordinateVariable,
    MissingVariable,
)


class TestCoordinateUtilities(TestCase):
    """A class for testing functions in the `hoss.coordinate_utilities`
    module.

    """

    @classmethod
    def setUpClass(cls):
        """Create fixtures that can be reused for all tests."""
        cls.config = config(validate=False)
        cls.logger = getLogger('tests')
        cls.varinfo = VarInfoFromDmr(
            'tests/data/SC_SPL3SMP_008.dmr',
            'SPL3SMP',
            config_file='hoss/hoss_config.json',
        )
        cls.test_varinfo = VarInfoFromDmr(
            'tests/data/SC_SPL3SMP_008_fake.dmr',
            'SPL3SMP',
            config_file='hoss/hoss_config.json',
        )
        cls.nc4file = 'tests/data/SC_SPL3SMP_008_prefetch.nc4'
        cls.latitude = '/Soil_Moisture_Retrieval_Data_AM/latitude'
        cls.longitude = '/Soil_Moisture_Retrieval_Data_AM/longitude'

        cls.lon_arr = np.array(
            [
                [-179.3, -120.2, -60.6, -9999, -9999, -9999, 80.2, 120.6, 150.5, 178.4],
                [-179.3, -120.2, -60.6, -999, 999, -9999, 80.2, 120.6, 150.5, 178.4],
                [-179.3, -120.2, -60.6, -9999, -9999, -9999, 80.2, 120.6, 150.5, 178.4],
                [-179.3, -120.2, -60.6, -9999, -9999, -9999, 80.2, 120.6, 150.5, 178.4],
                [-179.3, -120.2, -60.6, -9999, -9999, -9999, 80.2, 120.6, 150.5, 178.4],
            ]
        )

        cls.lat_arr = np.array(
            [
                [89.3, 89.3, -9999, 89.3, 89.3, 89.3, -9999, 89.3, 89.3, 89.3],
                [50.3, 50.3, 50.3, 50.3, 50.3, 50.3, -9999, 50.3, 50.3, 50.3],
                [1.3, 1.3, 1.3, 1.3, 1.3, 1.3, -9999, -9999, 1.3, 1.3],
                [-9999, -60.2, -60.2, -99, -9999, -9999, -60.2, -60.2, -60.2, -60.2],
                [-88.1, -88.1, -88.1, 99, -9999, -9999, -88.1, -88.1, -88.1, -88.1],
            ]
        )

    def setUp(self):
        """Create fixtures that should be unique per test."""

    def tearDown(self):
        """Remove per-test fixtures."""

    def test_get_coordinate_variables(self):
        """Ensure that the correct coordinate variables are
        retrieved for the reqquested science variable

        """
        requested_science_variables = [
            '/Soil_Moisture_Retrieval_Data_AM/surface_flag',
            '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm',
        ]
        expected_coordinate_variables = (
            [
                '/Soil_Moisture_Retrieval_Data_AM/latitude',
                '/Soil_Moisture_Retrieval_Data_PM/latitude_pm',
            ],
            [
                '/Soil_Moisture_Retrieval_Data_AM/longitude',
                '/Soil_Moisture_Retrieval_Data_PM/longitude_pm',
            ],
        )

        with self.subTest('Retrieves expected coordinates for the requested variables'):
            actual_coordinate_variables = get_coordinate_variables(
                self.varinfo, requested_science_variables
            )
            # the order of the results maybe random
            self.assertEqual(
                len(expected_coordinate_variables), len(actual_coordinate_variables)
            )
            self.assertCountEqual(
                expected_coordinate_variables[0], actual_coordinate_variables[0]
            )
            self.assertCountEqual(
                expected_coordinate_variables[0], actual_coordinate_variables[0]
            )
            for expected_variable in expected_coordinate_variables[0]:
                self.assertIn(expected_variable, actual_coordinate_variables[0])

            for expected_variable in expected_coordinate_variables[1]:
                self.assertIn(expected_variable, actual_coordinate_variables[1])

    def test_get_1d_dim_array_data_from_dimvalues(self):
        """Ensure that the dimension scale generated from the
        provided dimension values are accurate for ascending and
        descending scales
        """

        with self.subTest('valid ascending dim scale'):
            dim_values_asc = np.array([2, 4])
            dim_indices_asc = np.array([0, 1])
            dim_size_asc = 12
            expected_dim_asc = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
            dim_array_values = get_1d_dim_array_data_from_dimvalues(
                dim_values_asc, dim_indices_asc, dim_size_asc
            )
            self.assertTrue(np.array_equal(dim_array_values, expected_dim_asc))

        with self.subTest('valid descending dim scale'):
            dim_values_desc = np.array([100, 70])
            dim_indices_desc = np.array([2, 5])
            dim_size_desc = 10
            expected_dim_desc = np.array([120, 110, 100, 90, 80, 70, 60, 50, 40, 30])

            dim_array_values = get_1d_dim_array_data_from_dimvalues(
                dim_values_desc, dim_indices_desc, dim_size_desc
            )
            self.assertTrue(np.array_equal(dim_array_values, expected_dim_desc))

        with self.subTest('invalid dimension values'):
            dim_values_invalid = np.array([2, 2])
            dim_indices_asc = np.array([0, 1])
            dim_size_asc = 12
            with self.assertRaises(InvalidCoordinateData) as context:
                get_1d_dim_array_data_from_dimvalues(
                    dim_values_invalid, dim_indices_asc, dim_size_asc
                )
            self.assertEqual(
                context.exception.message,
                'The data does not have at least two valid values '
                'dim_value: "2" dim_index: "0"',
            )

        with self.subTest('invalid dimension indices'):
            dim_values_desc = np.array([100, 70])
            dim_indices_invalid = np.array([5, 5])
            dim_size_desc = 10
            with self.assertRaises(InvalidCoordinateData) as context:
                get_1d_dim_array_data_from_dimvalues(
                    dim_values_desc, dim_indices_invalid, dim_size_desc
                )
            self.assertEqual(
                context.exception.message,
                'The data does not have at least two valid values '
                'dim_value: "100" dim_index: "5"',
            )

    def test_get_coordinate_array(self):
        """Ensures that the expected lat/lon arrays are retrieved
        for the coordinate variables
        """
        expected_shape = (406, 964)
        with self.subTest('Expected latitude array'):
            with Dataset(self.nc4file, 'r') as prefetch_dataset:
                lat_prefetch_arr = get_coordinate_array(
                    prefetch_dataset,
                    self.latitude,
                )
                self.assertTupleEqual(lat_prefetch_arr.shape, expected_shape)
                np.testing.assert_array_equal(
                    lat_prefetch_arr, prefetch_dataset[self.latitude][:]
                )
        with self.subTest('Expected longitude array'):
            with Dataset(self.nc4file, 'r') as prefetch_dataset:
                lon_prefetch_arr = get_coordinate_array(
                    prefetch_dataset, self.longitude
                )
                self.assertTupleEqual(lon_prefetch_arr.shape, expected_shape)
                np.testing.assert_array_equal(
                    lon_prefetch_arr, prefetch_dataset[self.longitude][:]
                )
        with self.subTest('Missing coordinate'):
            with Dataset(self.nc4file, 'r') as prefetch_dataset:
                with self.assertRaises(MissingCoordinateVariable) as context:
                    coord_arr = (
                        get_coordinate_array(
                            prefetch_dataset,
                            '/Soil_Moisture_Retrieval_Data_AM/longitude_centroid',
                        ),
                    )
                    self.assertEqual(
                        context.exception.message,
                        'Coordinate: "/Soil_Moisture_Retrieval_Data_AM/latitude_centroid" is '
                        'not present in coordinate prefetch file.',
                    )

    def test_get_projected_dimension_names(self):
        """Ensure that the expected projected dimension name
        is returned for the coordinate variables
        """

        expected_projected_names = [
            '/Soil_Moisture_Retrieval_Data_AM/projected_y',
            '/Soil_Moisture_Retrieval_Data_AM/projected_x',
        ]

        with self.subTest(
            'Retrieves expected projected dimension names for a science variable'
        ):
            self.assertListEqual(
                get_projected_dimension_names(self.varinfo, self.latitude),
                expected_projected_names,
            )

        with self.subTest(
            'Retrieves expected dimension names for the longitude variable'
        ):
            self.assertEqual(
                get_projected_dimension_names(self.varinfo, self.longitude),
                expected_projected_names,
            )

        with self.subTest('Raises exception for missing coordinate variable'):
            with self.assertRaises(MissingVariable) as context:
                get_projected_dimension_names(
                    self.varinfo, '/Soil_Moisture_Retrieval_Data_AM/random_variable'
                )
            self.assertEqual(
                context.exception.message,
                '"/Soil_Moisture_Retrieval_Data_AM/random_variable" is '
                'not present in source granule file.',
            )

    def test_get_projected_dimension_names_from_coordinate_variables(self):
        """Ensure that the expected projected dimension name
        is returned for the coordinate variables
        """

        expected_override_dimensions_AM = [
            '/Soil_Moisture_Retrieval_Data_AM/projected_y',
            '/Soil_Moisture_Retrieval_Data_AM/projected_x',
        ]
        expected_override_dimensions_PM = [
            '/Soil_Moisture_Retrieval_Data_PM/projected_y',
            '/Soil_Moisture_Retrieval_Data_PM/projected_x',
        ]

        with self.subTest(
            'Retrieves expected override dimensions for the science variable'
        ):
            self.assertListEqual(
                get_projected_dimension_names_from_coordinate_variables(
                    self.varinfo, '/Soil_Moisture_Retrieval_Data_AM/surface_flag'
                ),
                expected_override_dimensions_AM,
            )

        with self.subTest(
            'Retrieves expected override dimensions for the longitude variable'
        ):
            self.assertListEqual(
                get_projected_dimension_names_from_coordinate_variables(
                    self.varinfo, self.longitude
                ),
                expected_override_dimensions_AM,
            )

        with self.subTest(
            'Retrieves expected override dimensions for the latitude variable'
        ):
            self.assertListEqual(
                get_projected_dimension_names_from_coordinate_variables(
                    self.varinfo, self.latitude
                ),
                expected_override_dimensions_AM,
            )

        with self.subTest(
            'Retrieves expected override dimensions science variable with a different grid'
        ):
            self.assertListEqual(
                get_projected_dimension_names_from_coordinate_variables(
                    self.varinfo, '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm'
                ),
                expected_override_dimensions_PM,
            )
        with self.subTest(
            'Retrieves empty dimensions list when science variable has no coordinates'
        ):
            self.assertListEqual(
                get_projected_dimension_names_from_coordinate_variables(
                    self.varinfo, '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm'
                ),
                expected_override_dimensions_PM,
            )

    def test_get_row_col_sizes_from_coordinate_datasets(self):
        """Ensure that the correct row and column sizes are
        returned for the requested coordinates
        """

        with self.subTest('Retrieves the expected row col sizes from the coordinates'):
            expected_row_col_sizes = (5, 10)
            self.assertEqual(
                get_row_col_sizes_from_coordinate_datasets(self.lat_arr, self.lon_arr),
                expected_row_col_sizes,
            )
        with self.subTest('Retrieves the expected row col sizes for the dim array'):
            self.assertEqual(
                get_row_col_sizes_from_coordinate_datasets(
                    np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8, 9])
                ),
                (4, 5),
            )
        with self.subTest(
            'Raises an exception when the lat and lon array shapes do not match'
        ):
            lat_mismatched_array = np.array([[1, 2, 3], [3, 4, 5]])
            lon_mismatched_array = np.array([[6, 7], [8, 9], [10, 11]])
            with self.assertRaises(IncompatibleCoordinateVariables) as context:
                get_row_col_sizes_from_coordinate_datasets(
                    lat_mismatched_array, lon_mismatched_array
                )
            self.assertEqual(
                context.exception.message,
                f'Longitude coordinate shape: "{lon_mismatched_array.shape}"'
                f'does not match the latitude coordinate shape: "{lat_mismatched_array.shape}"',
            )
        with self.subTest(
            'Raises an exception when Both arrays are 1-D, but latitude has a zero size'
        ):
            lat_empty_size_array = np.array([])
            with self.assertRaises(IncompatibleCoordinateVariables) as context:
                get_row_col_sizes_from_coordinate_datasets(
                    lat_empty_size_array, np.array([5, 6, 7, 8])
                )

        with self.subTest(
            'Raises an exception when Both arrays are 1-D, but longitude has a zero size'
        ):
            lon_empty_size_array = np.array([])
            with self.assertRaises(IncompatibleCoordinateVariables) as context:
                get_row_col_sizes_from_coordinate_datasets(
                    np.array([6, 7, 8, 9]), lon_empty_size_array
                )

        with self.subTest(
            'Raises an exception when latitude array that is zero dimensional'
        ):
            lat_empty_ndim_array = np.array(
                0,
            )
            with self.assertRaises(IncompatibleCoordinateVariables) as context:
                get_row_col_sizes_from_coordinate_datasets(
                    lat_empty_ndim_array, np.array([1, 2, 3, 4])
                )

        with self.subTest(
            'Raises an exception when longitude array that is zero dimensional'
        ):
            lon_empty_ndim_array = np.array(
                0,
            )
            with self.assertRaises(IncompatibleCoordinateVariables) as context:
                get_row_col_sizes_from_coordinate_datasets(
                    np.array([1, 2, 3, 4]), lon_empty_ndim_array
                )

    def test_get_valid_indices(self):
        """Ensure that latitude and longitude values are correctly identified as
        ascending or descending.

        """
        expected_valid_indices_lon_arr_over_range = np.array([0, 1, 2, 6, 7, 8, 9])

        fill_array = np.array([-9999.0, -9999.0, -9999.0, -9999.0])

        with self.subTest('valid indices for latitude with fill values'):
            expected_valid_indices_lat_arr_with_fill = np.array(
                [False, True, True, True, True]
            )
            valid_indices_lat_arr = get_valid_indices(
                self.lat_arr[:, 2], self.varinfo.get_variable(self.latitude)
            )
            np.testing.assert_array_equal(
                valid_indices_lat_arr, expected_valid_indices_lat_arr_with_fill
            )
        with self.subTest('valid indices for longitude with fill values'):
            expected_valid_indices_lon_arr_with_fill = np.array(
                [True, True, True, False, False, False, True, True, True, True]
            )
            valid_indices_lon_arr = get_valid_indices(
                self.lon_arr[0, :], self.varinfo.get_variable(self.longitude)
            )
            np.testing.assert_array_equal(
                valid_indices_lon_arr, expected_valid_indices_lon_arr_with_fill
            )
        with self.subTest('latitude values beyond valid range'):
            expected_valid_indices_lat_arr_over_range = np.array(
                [True, True, True, False, False]
            )
            valid_indices_lat_arr = get_valid_indices(
                self.lat_arr[:, 3], self.varinfo.get_variable(self.latitude)
            )
            np.testing.assert_array_equal(
                valid_indices_lat_arr, expected_valid_indices_lat_arr_over_range
            )
        with self.subTest('longitude values beyond valid range'):
            expected_valid_indices_lon_arr_over_range = np.array(
                [True, True, True, False, False, False, True, True, True, True]
            )
            valid_indices_lon_arr = get_valid_indices(
                self.lon_arr[1, :], self.varinfo.get_variable(self.longitude)
            )
            np.testing.assert_array_equal(
                valid_indices_lon_arr, expected_valid_indices_lon_arr_over_range
            )
        with self.subTest('all fill values - no valid indices'):
            expected_valid_indices_fill_values = np.array([False, False, False, False])
            valid_indices_all_fill = get_valid_indices(
                fill_array, self.varinfo.get_variable(self.longitude)
            )
            np.testing.assert_array_equal(
                valid_indices_all_fill, expected_valid_indices_fill_values
            )

    def test_get_variables_with_anonymous_dims(self):
        """Ensure that variables with no dimensions are
        retrieved for the requested science variable

        """

        with self.subTest('Retrieves variables with no dimensions'):
            requested_science_variables = {
                '/Soil_Moisture_Retrieval_Data_AM/surface_flag',
                '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm',
            }
            variables_with_anonymous_dims = get_variables_with_anonymous_dims(
                self.test_varinfo, requested_science_variables
            )
            self.assertSetEqual(
                variables_with_anonymous_dims,
                requested_science_variables,
            )
        with self.subTest('Does not retrieve variables with dimensions'):
            variables_with_anonymous_dims = get_variables_with_anonymous_dims(
                self.test_varinfo, {'/Soil_Moisture_Retrieval_Data_AM/variable_has_dim'}
            )
            self.assertTrue(len(variables_with_anonymous_dims) == 0)

        with self.subTest(
            'Only retrieves variables with anonymous dimensions,'
            'when the request has both'
        ):
            requested_science_variables_with_dimensions = {
                '/Soil_Moisture_Retrieval_Data_AM/variable_has_dim',
                '/Soil_Moisture_Retrieval_Data_AM/variable_has_anonymous_dim',
            }
            variables_with_anonymous_dims = get_variables_with_anonymous_dims(
                self.test_varinfo, requested_science_variables_with_dimensions
            )
            self.assertSetEqual(
                variables_with_anonymous_dims,
                {'/Soil_Moisture_Retrieval_Data_AM/variable_has_anonymous_dim'},
            )
        with self.subTest(
            'retrieves variables with fake dimensions,' 'when the request has both'
        ):
            variables_with_fake_dims = get_variables_with_anonymous_dims(
                self.test_varinfo,
                {'/Soil_Moisture_Retrieval_Data_PM/variable_with_fake_dims'},
            )
            self.assertSetEqual(
                variables_with_fake_dims,
                {'/Soil_Moisture_Retrieval_Data_PM/variable_with_fake_dims'},
            )

    def test_any_absent_dimension_variables(self):
        """Ensure that variables with fake dimensions are
        detected with a True return value

        """

        with self.subTest('Returns true for variables with fake dimensions'):
            variable_has_fake_dims = any_absent_dimension_variables(
                self.test_varinfo,
                '/Soil_Moisture_Retrieval_Data_PM/variable_with_fake_dims',
            )
            self.assertTrue(variable_has_fake_dims)
        with self.subTest('Returns false for variables with dimensions'):
            variable_has_fake_dims = any_absent_dimension_variables(
                self.test_varinfo, '/Soil_Moisture_Retrieval_Data_AM/variable_has_dim'
            )
            self.assertFalse(variable_has_fake_dims)

    def test_get_max_x_spread_pts(self):
        """Ensure that two valid sets of indices are returned by the function
        with a masked dataset as input

        """

        with self.subTest('Get two sets of valid indices for points from coordinates'):
            valid_values = np.array(
                [
                    [True, True, True, True, False, False, True, True, True, True],
                    [True, True, True, False, False, False, True, True, True, True],
                    [True, True, True, False, True, False, True, True, True, True],
                    [True, True, True, False, False, False, True, True, True, True],
                    [True, True, False, False, False, False, True, True, True, True],
                ]
            )
            expected_indices = [[0, 0], [0, 9]]

            actual_indices = get_max_x_spread_pts(~valid_values)

            self.assertTrue(actual_indices[0] == expected_indices[0])
            self.assertTrue(actual_indices[1] == expected_indices[1])

    def test_get_valid_row_col_pairs(self):
        """Ensure that two sets of valid indices are
        returned by the method with a set of lat/lon coordinates as input

        """
        with self.subTest('Get two sets of valid indices from coordinates dataset'):
            expected_grid_indices = (
                [(0, 0), (0, 9)],
                [(0, 0), (4, 0)],
            )
            actual_grid_indices = get_valid_row_col_pairs(
                self.lat_arr,
                self.lon_arr,
                self.varinfo.get_variable(self.latitude),
                self.varinfo.get_variable(self.longitude),
            )

            self.assertTrue(actual_grid_indices[0], expected_grid_indices[0])
            self.assertTrue(actual_grid_indices[1], expected_grid_indices[1])

    def test_get_x_y_values_from_geographic_points(self):
        """Ensure that the correct x-values are returned by the function
        with a set of geographic coordinates as input.

        """
        crs = CRS.from_cf(
            {
                'false_easting': 0.0,
                'false_northing': 0.0,
                'longitude_of_central_meridian': 0.0,
                'standard_parallel': 30.0,
                'grid_mapping_name': 'lambert_cylindrical_equal_area',
            }
        )

        with self.subTest('Get valid x-y points from coordinates in a row'):
            two_col_points_in_a_row = [(-179.3, 89.3), (178.4, 89.3)]

            expected_x_y_points = [
                (-17299990.048985746, 7341677.255608977),
                (17213152.396759935, 7341677.255608977),
            ]
            actual_x_y_points = get_x_y_values_from_geographic_points(
                two_col_points_in_a_row, crs
            )
            self.assertListEqual(actual_x_y_points, expected_x_y_points)

        with self.subTest('Get valid x-y points from coordinates in a col'):
            two_row_points_in_a_col = (-179.3, 89.3), (-179.3, -88.1)
            expected_x_y_points = [
                (-17299990.048985746, 7341677.255608977),
                (-17299990.048985746, -7338157.219843731),
            ]
            actual_x_y_points = get_x_y_values_from_geographic_points(
                two_row_points_in_a_col, crs
            )
            self.assertListEqual(actual_x_y_points, expected_x_y_points)

    def test_get_dimension_array_from_geo_points(self):
        """Ensure that a valid x/y dimension array is returned
        with a geo array of lat/lon values

        """
        crs = CRS.from_cf(
            {
                'false_easting': 0.0,
                'false_northing': 0.0,
                'longitude_of_central_meridian': 0.0,
                'standard_parallel': 30.0,
                'grid_mapping_name': 'lambert_cylindrical_equal_area',
            }
        )
        with self.subTest('Get y dimension array from geo coordinates'):
            row_indices = [[0, 0], [4, 0]]
            ymax = 7341677.255608977
            ymin = -25687950.314159617

            dim_array = get_dimension_array_from_geo_points(
                self.lat_arr, self.lon_arr, crs, row_indices, 10, True
            )
            self.assertEqual(dim_array.size, 10)
            self.assertEqual(dim_array[0], ymax)
            self.assertEqual(dim_array[-1], ymin)

        with self.subTest('Get x dimension array from geo coordinates'):
            col_indices = [[0, 0], [0, 9]]
            xmin = -17299990.048985746
            xmax = -1960815.628654331
            dim_array = get_dimension_array_from_geo_points(
                self.lat_arr, self.lon_arr, crs, col_indices, 5, False
            )
            self.assertEqual(dim_array.size, 5)
            self.assertEqual(dim_array[0], xmin)
            self.assertEqual(dim_array[-1], xmax)
