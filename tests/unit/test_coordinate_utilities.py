from logging import getLogger
from os.path import exists
from shutil import copy, rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import ANY, patch

import numpy as np
from harmony.util import config
from netCDF4 import Dataset
from numpy.testing import assert_array_equal
from varinfo import VarInfoFromDmr

from hoss.coordinate_utilities import (
    get_coordinate_array,
    get_coordinate_variables,
    get_dimension_scale_from_dimvalues,
    get_fill_value_for_coordinate,
    get_projected_dimension_names,
    get_projected_dimension_names_from_coordinate_variables,
    get_row_col_sizes_from_coordinate_datasets,
    get_valid_indices,
    get_variables_with_anonymous_dims,
)
from hoss.exceptions import (
    InvalidCoordinateDataset,
    InvalidCoordinateVariable,
    InvalidSizingInCoordinateVariables,
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
        cls.nc4file = 'tests/data/SC_SPL3SMP_008_prefetch.nc4'
        cls.latitude = '/Soil_Moisture_Retrieval_Data_AM/latitude'
        cls.longitude = '/Soil_Moisture_Retrieval_Data_AM/longitude'

        cls.lon_arr = np.array(
            [
                [-179.3, -120.2, -60.6, -9999, -9999, -9999, 80.2, 120.6, 150.5, 178.4],
                [-179.3, -120.2, -60.6, -9999, -9999, -9999, 80.2, 120.6, 150.5, 178.4],
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
                [-9999, -60.2, -60.2, -9999, -9999, -9999, -60.2, -60.2, -60.2, -60.2],
                [-88.1, -88.1, -88.1, -9999, -9999, -9999, -88.1, -88.1, -88.1, -88.1],
            ]
        )

        cls.lon_arr_reversed = np.array(
            [
                [
                    -179.3,
                    -179.3,
                    -179.3,
                    -179.3,
                    -9999,
                    -9999,
                    -179.3,
                    -179.3,
                    -179.3,
                    -179.3,
                ],
                [
                    -120.2,
                    -120.2,
                    -120.2,
                    -9999,
                    -9999,
                    -120.2,
                    -120.2,
                    -120.2,
                    -120.2,
                    -120.2,
                ],
                [20.6, 20.6, 20.6, 20.6, 20.6, 20.6, 20.6, 20.6, -9999, -9999],
                [150.5, 150.5, 150.5, 150.5, 150.5, 150.5, -9999, -9999, 150.5, 150.5],
                [178.4, 178.4, 178.4, 178.4, 178.4, 178.4, 178.4, -9999, 178.4, 178.4],
            ]
        )

        cls.lat_arr_reversed = np.array(
            [
                [89.3, 79.3, -9999, 59.3, 29.3, 2.1, -9999, -59.3, -79.3, -89.3],
                [89.3, 79.3, 60.3, 59.3, 29.3, 2.1, -9999, -59.3, -79.3, -89.3],
                [89.3, -9999, 60.3, 59.3, 29.3, 2.1, -9999, -9999, -9999, -89.3],
                [-9999, 79.3, -60.3, -9999, -9999, -9999, -60.2, -59.3, -79.3, -89.3],
                [-89.3, 79.3, -60.3, -9999, -9999, -9999, -60.2, -59.3, -79.3, -9999],
            ]
        )

    def setUp(self):
        """Create fixtures that should be unique per test."""

    def tearDown(self):
        """Remove per-test fixtures."""

    def test_create_dimension_scales_from_coordinates(self):
        """Ensure that the dimension scales created from the
        coordinates are as expected
        """

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

        with self.subTest('Retrieves expected coordinates for the requested variable'):
            actual_coordinate_variables = get_coordinate_variables(
                self.varinfo, requested_science_variables
            )
            print(f'expected_coordinate_variables={expected_coordinate_variables}')
            print(f'actual_coordinate_variables={actual_coordinate_variables}')
            self.maxDiff = None
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

    def test_get_dimension_scale_from_dimvalues(self):
        """Ensure that the dimension scale generated from the
        provided dimension values are accurate for ascending and
        descending scales
        """

        dim_values_asc = np.array([2, 4])
        dim_indices_asc = np.array([0, 1])
        dim_size_asc = 12
        expected_dim_asc = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])

        dim_values_desc = np.array([100, 70])
        dim_indices_desc = np.array([2, 5])
        dim_size_desc = 10
        expected_dim_desc = np.array([120, 110, 100, 90, 80, 70, 60, 50, 40, 30])

        dim_values_invalid = np.array([2, 2])
        dim_indices_asc = np.array([0, 1])
        dim_size_asc = 12

        dim_values_desc = np.array([100, 70])
        dim_indices_invalid = np.array([5, 5])
        dim_size_desc = 10

        with self.subTest('valid ascending dim scale'):
            dim_scale_values = get_dimension_scale_from_dimvalues(
                dim_values_asc, dim_indices_asc, dim_size_asc
            )
            self.assertTrue(np.array_equal(dim_scale_values, expected_dim_asc))

        with self.subTest('valid descending dim scale'):
            dim_scale_values = get_dimension_scale_from_dimvalues(
                dim_values_desc, dim_indices_desc, dim_size_desc
            )
            self.assertTrue(np.array_equal(dim_scale_values, expected_dim_desc))

        with self.subTest('invalid dimension values'):
            with self.assertRaises(InvalidCoordinateDataset) as context:
                get_dimension_scale_from_dimvalues(
                    dim_values_invalid, dim_indices_asc, dim_size_asc
                )
            self.assertEqual(
                context.exception.message,
                'Cannot compute the dimension resolution for '
                'dim_value: "2" dim_index: "0"',
            )

        with self.subTest('invalid dimension indices'):
            with self.assertRaises(InvalidCoordinateDataset) as context:
                get_dimension_scale_from_dimvalues(
                    dim_values_desc, dim_indices_invalid, dim_size_desc
                )
            self.assertEqual(
                context.exception.message,
                'Cannot compute the dimension resolution for '
                'dim_value: "100" dim_index: "5"',
            )

    def test_get_fill_value_for_coordinate(self):
        """Ensure that the fill values for the coordinates
        are correctly returned

        """
        latitude_coordinate = self.varinfo.get_variable(self.latitude)
        with self.subTest('Retrieves expected fill value for the latitude variable'):
            self.assertEqual(
                get_fill_value_for_coordinate(latitude_coordinate), -9999.0
            )
        with self.subTest('Returns None when there is no fill value for a variable'):
            self.assertEqual(
                get_fill_value_for_coordinate(
                    self.varinfo.get_variable(
                        '/Soil_Moisture_Retrieval_Data_AM/tb_time_utc'
                    )
                ),
                None,
            )

    def test_get_coordinate_array(self):
        """Ensures that the expected lat/lon arrays are retrieved
        for the coordinate variables
        """
        prefetch_dataset = Dataset(self.nc4file, 'r')
        expected_shape = (406, 964)
        with self.subTest('Expected latitude array'):
            lat_prefetch_arr = get_coordinate_array(
                prefetch_dataset,
                self.varinfo.get_variable(self.latitude).full_name_path,
            )
            self.assertTupleEqual(lat_prefetch_arr.shape, expected_shape)

        with self.subTest('Expected longitude array'):
            lon_prefetch_arr = get_coordinate_array(
                prefetch_dataset,
                self.varinfo.get_variable(self.longitude).full_name_path,
            )
            self.assertTupleEqual(lon_prefetch_arr.shape, expected_shape)

        with self.subTest('Missing coordinate'):
            with self.assertRaises(MissingCoordinateVariable) as context:
                lat_prefetch_arr, lon_prefetch_arr = get_coordinate_array(
                    prefetch_dataset,
                    self.varinfo.get_variable(
                        '/Soil_Moisture_Retrieval_Data_AM/longitude_centroid'
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

    def test_get_projected_dimensions_names_from_coordinate_variables(self):
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
            self.assertEqual(
                get_projected_dimension_names_from_coordinate_variables(
                    self.varinfo, '/Soil_Moisture_Retrieval_Data_AM/surface_flag'
                ),
                expected_override_dimensions_AM,
            )

        with self.subTest(
            'Retrieves expected override dimensions for the longitude variable'
        ):
            self.assertEqual(
                get_projected_dimension_names_from_coordinate_variables(
                    self.varinfo, self.longitude
                ),
                expected_override_dimensions_AM,
            )

        with self.subTest(
            'Retrieves expected override dimensions for the latitude variable'
        ):
            self.assertEqual(
                get_projected_dimension_names_from_coordinate_variables(
                    self.varinfo, self.latitude
                ),
                expected_override_dimensions_AM,
            )

        with self.subTest(
            'Retrieves expected override dimensions science variable with a different grid'
        ):
            self.assertEqual(
                get_projected_dimension_names_from_coordinate_variables(
                    self.varinfo, '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm'
                ),
                expected_override_dimensions_PM,
            )

    def test_get_row_col_sizes_from_coordinate_datasets(self):
        """Ensure that the correct row and column sizes are
        returned for the requested coordinates
        """
        expected_row_col_sizes = (5, 10)
        lat_1d_array = np.array([1, 2, 3, 4])
        lon_1d_array = np.array([6, 7, 8, 9])
        lat_irregular_array = np.array([[1, 2, 3], [3, 4, 5]])
        lon_irregular_array = np.array([[6, 7], [8, 9], [10, 11]])
        with self.subTest('Retrieves the expected row col sizes from the coordinates'):
            self.assertEqual(
                get_row_col_sizes_from_coordinate_datasets(self.lat_arr, self.lon_arr),
                expected_row_col_sizes,
            )
        with self.subTest('Retrieves the expected row col sizes for the 1d array'):
            self.assertEqual(
                get_row_col_sizes_from_coordinate_datasets(lat_1d_array, lon_1d_array),
                (4, 4),
            )

        with self.subTest(
            'Raises an exception when the lat and lon array shapes do not match'
        ):
            with self.assertRaises(InvalidSizingInCoordinateVariables) as context:
                get_row_col_sizes_from_coordinate_datasets(
                    lat_irregular_array, lon_irregular_array
                )
            self.assertEqual(
                context.exception.message,
                f'Longitude coordinate shape: "{lon_irregular_array.shape}"'
                f'does not match the latitude coordinate shape: "{lat_irregular_array.shape}"',
            )

    def test_get_valid_indices(self):
        """Ensure that latitude and longitude values are correctly identified as
        ascending or descending.

        """
        expected_valid_indices_lat_arr = np.array([0, 1, 2, 3, 4])
        expected_valid_indices_lon_arr = np.array([0, 1, 2, 6, 7, 8, 9])

        fill_array = np.array([-9999.0, -9999.0, -9999.0, -9999.0])
        with self.subTest('valid indices with no fill values configured'):
            valid_indices_lat_arr = get_valid_indices(
                self.lat_arr[:, -1], None, 'latitude'
            )
            self.assertTrue(
                np.array_equal(valid_indices_lat_arr, expected_valid_indices_lat_arr)
            )

        with self.subTest('valid indices with fill values configured'):
            valid_indices_lon_arr = get_valid_indices(
                self.lon_arr[0, :], -9999.0, 'longitude'
            )
            self.assertTrue(
                np.array_equal(valid_indices_lon_arr, expected_valid_indices_lon_arr)
            )

        with self.subTest('all fill values - no valid indices'):
            valid_indices = get_valid_indices(fill_array, -9999.0, 'longitude')
            self.assertTrue(valid_indices.size == 0)

    def test_get_variables_with_anonymous_dims(self):
        """Ensure that variables with no dimensions are
        retrieved for the reqquested science variable

        """
        requested_science_variables = {
            '/Soil_Moisture_Retrieval_Data_AM/surface_flag',
            '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm',
        }

        with self.subTest('Retrieves variables with no dimensions'):
            variables_with_anonymous_dims = get_variables_with_anonymous_dims(
                self.varinfo, requested_science_variables
            )
            self.assertSetEqual(
                variables_with_anonymous_dims,
                requested_science_variables,
            )
