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
    create_dimension_arrays_from_coordinates,
    get_2d_coordinate_array,
    get_coordinate_variables,
    get_dimension_array_names,
    get_dimension_array_names_from_coordinates,
    get_dimension_order_and_dim_values,
    get_max_spread_pts,
    get_row_col_sizes_from_coordinates,
    get_valid_indices,
    get_valid_sample_pts,
    get_variables_with_anonymous_dims,
    interpolate_dim_values_from_sample_pts,
)
from hoss.exceptions import (
    IncompatibleCoordinateVariables,
    InvalidCoordinateData,
    InvalidCoordinateVariable,
    MissingCoordinateVariable,
    MissingVariable,
    UnsupportedDimensionOrder,
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
        cls.lon_arr_3d = np.array(
            [
                [
                    [
                        -179.3,
                        -120.2,
                        -60.6,
                        -9999,
                        -9999,
                        -9999,
                        80.2,
                        120.6,
                        150.5,
                        178.4,
                    ],
                    [
                        -179.3,
                        -120.2,
                        -60.6,
                        -999,
                        999,
                        -9999,
                        80.2,
                        120.6,
                        150.5,
                        178.4,
                    ],
                    [
                        -179.3,
                        -120.2,
                        -60.6,
                        -9999,
                        -9999,
                        -9999,
                        80.2,
                        120.6,
                        150.5,
                        178.4,
                    ],
                    [
                        -179.3,
                        -120.2,
                        -60.6,
                        -9999,
                        -9999,
                        -9999,
                        80.2,
                        120.6,
                        150.5,
                        178.4,
                    ],
                    [
                        -179.3,
                        -120.2,
                        -60.6,
                        -9999,
                        -9999,
                        -9999,
                        80.2,
                        120.6,
                        150.5,
                        178.4,
                    ],
                ],
                [
                    [
                        -179.3,
                        -120.2,
                        -60.6,
                        -9999,
                        -9999,
                        -9999,
                        80.2,
                        120.6,
                        150.5,
                        178.4,
                    ],
                    [
                        -179.3,
                        -120.2,
                        -60.6,
                        -999,
                        999,
                        -9999,
                        80.2,
                        120.6,
                        150.5,
                        178.4,
                    ],
                    [
                        -179.3,
                        -120.2,
                        -60.6,
                        -9999,
                        -9999,
                        -9999,
                        80.2,
                        120.6,
                        150.5,
                        178.4,
                    ],
                    [
                        -179.3,
                        -120.2,
                        -60.6,
                        -9999,
                        -9999,
                        -9999,
                        80.2,
                        120.6,
                        150.5,
                        178.4,
                    ],
                    [
                        -179.3,
                        -120.2,
                        -60.6,
                        -9999,
                        -9999,
                        -9999,
                        80.2,
                        120.6,
                        150.5,
                        178.4,
                    ],
                ],
            ]
        )

        cls.lat_arr_3d = np.array(
            [
                [
                    [89.3, 89.3, -9999, 89.3, 89.3, 89.3, -9999, 89.3, 89.3, 89.3],
                    [50.3, 50.3, 50.3, 50.3, 50.3, 50.3, -9999, 50.3, 50.3, 50.3],
                    [1.3, 1.3, 1.3, 1.3, 1.3, 1.3, -9999, -9999, 1.3, 1.3],
                    [
                        -9999,
                        -60.2,
                        -60.2,
                        -99,
                        -9999,
                        -9999,
                        -60.2,
                        -60.2,
                        -60.2,
                        -60.2,
                    ],
                    [-88.1, -88.1, -88.1, 99, -9999, -9999, -88.1, -88.1, -88.1, -88.1],
                ],
                [
                    [89.3, 89.3, -9999, 89.3, 89.3, 89.3, -9999, 89.3, 89.3, 89.3],
                    [50.3, 50.3, 50.3, 50.3, 50.3, 50.3, -9999, 50.3, 50.3, 50.3],
                    [1.3, 1.3, 1.3, 1.3, 1.3, 1.3, -9999, -9999, 1.3, 1.3],
                    [
                        -9999,
                        -60.2,
                        -60.2,
                        -99,
                        -9999,
                        -9999,
                        -60.2,
                        -60.2,
                        -60.2,
                        -60.2,
                    ],
                    [-88.1, -88.1, -88.1, 99, -9999, -9999, -88.1, -88.1, -88.1, -88.1],
                ],
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
                [
                    150.5,
                    150.5,
                    150.5,
                    150.5,
                    150.5,
                    150.5,
                    -9999,
                    -9999,
                    150.5,
                    150.5,
                ],
                [
                    178.4,
                    178.4,
                    178.4,
                    178.4,
                    178.4,
                    178.4,
                    178.4,
                    -9999,
                    178.4,
                    178.4,
                ],
            ]
        )
        cls.lat_arr_reversed = np.array(
            [
                [89.3, 79.3, -9999, 59.3, 29.3, 2.1, -9999, -59.3, -79.3, -89.3],
                [89.3, 79.3, 60.3, 59.3, 29.3, 2.1, -9999, -59.3, -79.3, -89.3],
                [89.3, -9999, 60.3, 59.3, 29.3, 2.1, -9999, -9999, -9999, -89.3],
                [
                    -9999,
                    79.3,
                    -60.3,
                    -9999,
                    -9999,
                    -9999,
                    -60.2,
                    -59.3,
                    -79.3,
                    -89.3,
                ],
                [
                    89.3,
                    79.3,
                    -60.3,
                    -9999,
                    -9999,
                    -9999,
                    -60.2,
                    -59.3,
                    -79.3,
                    -9999,
                ],
            ]
        )
        cls.smap_ftp_varinfo = VarInfoFromDmr(
            'tests/data/SC_SPL3FTP_004.dmr',
            'SPL3FTP',
            'hoss/hoss_config.json',
        )
        cls.smap_ftp_file_path = 'tests/data/SC_SPL3FTP_004_prefetch.nc4'

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

        with self.subTest('No lat coordinate variables for the requested variables'):
            # should return one valid list and an empty list
            actual_coordinate_variables = get_coordinate_variables(
                self.test_varinfo,
                ['/Soil_Moisture_Retrieval_Data_AM/no_lat_coordinate_variable'],
            )
            self.assertTupleEqual(
                actual_coordinate_variables,
                ([], ['/Soil_Moisture_Retrieval_Data_AM/longitude']),
            )
        with self.subTest('No lon coordinate variables for the requested variables'):
            # should return one valid list and an empty list
            actual_coordinate_variables = get_coordinate_variables(
                self.test_varinfo,
                ['/Soil_Moisture_Retrieval_Data_AM/no_lon_coordinate_variable'],
            )
            self.assertTupleEqual(
                actual_coordinate_variables,
                (['/Soil_Moisture_Retrieval_Data_AM/latitude'], []),
            )
        with self.subTest('No coordinate variables for the requested variables'):
            # should return empty lists
            actual_coordinate_variables = get_coordinate_variables(
                self.test_varinfo,
                ['/Soil_Moisture_Retrieval_Data_AM/no_coordinate_variable'],
            )
            self.assertTupleEqual(actual_coordinate_variables, ([], []))
        with self.subTest('Missing coordinate variables'):
            # should return empty lists
            missing_coordinate_variables = get_coordinate_variables(
                self.test_varinfo,
                ['/Soil_Moisture_Retrieval_Data_AM/variable_with_missing_coordinates'],
            )
            self.assertTupleEqual(missing_coordinate_variables, ([], []))
        with self.subTest('Fake coordinate variables'):
            # should return empty lists
            fake_coordinate_variables = get_coordinate_variables(
                self.test_varinfo,
                ['/Soil_Moisture_Retrieval_Data_AM/variable_with_fake_coordinates'],
            )
            self.assertTupleEqual(fake_coordinate_variables, ([], []))

    def test_interpolate_dim_values_from_sample_pts(self):
        """Ensure that the dimension scale generated from the
        provided dimension values are accurate for ascending and
        descending scales
        """

        with self.subTest('valid ascending dim scale'):
            dim_values_asc = np.array([2, 4])
            dim_indices_asc = np.array([0, 1])
            dim_size_asc = 12
            expected_dim_asc = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
            dim_array_values = interpolate_dim_values_from_sample_pts(
                dim_values_asc, dim_indices_asc, dim_size_asc
            )
            self.assertTrue(np.array_equal(dim_array_values, expected_dim_asc))

        with self.subTest('valid descending dim scale'):
            dim_values_desc = np.array([100, 70])
            dim_indices_desc = np.array([2, 5])
            dim_size_desc = 10
            expected_dim_desc = np.array([120, 110, 100, 90, 80, 70, 60, 50, 40, 30])

            dim_array_values = interpolate_dim_values_from_sample_pts(
                dim_values_desc, dim_indices_desc, dim_size_desc
            )
            self.assertTrue(np.array_equal(dim_array_values, expected_dim_desc))

        with self.subTest('invalid dimension values'):
            dim_values_invalid = np.array([2, 2])
            dim_indices_asc = np.array([0, 1])
            dim_size_asc = 12
            with self.assertRaises(InvalidCoordinateData) as context:
                interpolate_dim_values_from_sample_pts(
                    dim_values_invalid, dim_indices_asc, dim_size_asc
                )
            self.assertEqual(
                context.exception.message,
                'No distinct valid coordinate points - ' 'dim_index=0, dim_value=2',
            )

        with self.subTest('invalid dimension indices'):
            dim_values_desc = np.array([100, 70])
            dim_indices_invalid = np.array([5, 5])
            dim_size_desc = 10
            with self.assertRaises(InvalidCoordinateData) as context:
                interpolate_dim_values_from_sample_pts(
                    dim_values_desc, dim_indices_invalid, dim_size_desc
                )
            self.assertEqual(
                context.exception.message,
                'No distinct valid coordinate points - ' 'dim_index=5, dim_value=100',
            )

    def test_get_2d_coordinate_array(self):
        """Ensures that the expected lat/lon arrays are retrieved
        for the coordinate variables
        """
        expected_shape = (406, 964)
        with self.subTest('Expected latitude array'):
            with Dataset(self.nc4file, 'r') as prefetch_dataset:
                lat_prefetch_arr = get_2d_coordinate_array(
                    prefetch_dataset,
                    self.latitude,
                )
                self.assertTupleEqual(lat_prefetch_arr.shape, expected_shape)
                np.testing.assert_array_equal(
                    lat_prefetch_arr, prefetch_dataset[self.latitude][:]
                )
        with self.subTest('Expected longitude array'):
            with Dataset(self.nc4file, 'r') as prefetch_dataset:
                lon_prefetch_arr = get_2d_coordinate_array(
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
                        get_2d_coordinate_array(
                            prefetch_dataset,
                            '/Soil_Moisture_Retrieval_Data_AM/longitude_centroid',
                        ),
                    )
                    self.assertEqual(
                        context.exception.message,
                        'Coordinate: "/Soil_Moisture_Retrieval_Data_AM/latitude_centroid" is '
                        'not present in coordinate prefetch file.',
                    )

    def test_get_dimension_array_names_from_coordinates(self):
        """Ensure that the expected projected dimension name
        is returned for the coordinate variables
        """

        expected_dimension_names = [
            '/Soil_Moisture_Retrieval_Data_AM/dim_y',
            '/Soil_Moisture_Retrieval_Data_AM/dim_x',
        ]

        with self.subTest(
            'Retrieves expected projected dimension names for a science variable'
        ):
            self.assertListEqual(
                get_dimension_array_names_from_coordinates(self.varinfo, self.latitude),
                expected_dimension_names,
            )

        with self.subTest(
            'Retrieves expected dimension names for the longitude variable'
        ):
            self.assertEqual(
                get_dimension_array_names_from_coordinates(
                    self.varinfo, self.longitude
                ),
                expected_dimension_names,
            )

        with self.subTest('Raises exception for missing coordinate variable'):
            with self.assertRaises(MissingVariable) as context:
                get_dimension_array_names_from_coordinates(
                    self.varinfo, '/Soil_Moisture_Retrieval_Data_AM/random_variable'
                )
            self.assertEqual(
                context.exception.message,
                '"/Soil_Moisture_Retrieval_Data_AM/random_variable" is '
                'not present in source granule file.',
            )

    def test_get_dimension_array_names(self):
        """Ensure that the expected projected dimension name
        is returned for the coordinate variables
        """

        expected_override_dimensions_AM = [
            '/Soil_Moisture_Retrieval_Data_AM/dim_y',
            '/Soil_Moisture_Retrieval_Data_AM/dim_x',
        ]
        expected_override_dimensions_PM = [
            '/Soil_Moisture_Retrieval_Data_PM/dim_y',
            '/Soil_Moisture_Retrieval_Data_PM/dim_x',
        ]

        with self.subTest(
            'Retrieves expected override dimensions for the science variable'
        ):
            self.assertListEqual(
                get_dimension_array_names(
                    self.varinfo, '/Soil_Moisture_Retrieval_Data_AM/surface_flag'
                ),
                expected_override_dimensions_AM,
            )

        with self.subTest(
            'Retrieves expected override dimensions for the longitude variable'
        ):
            self.assertListEqual(
                get_dimension_array_names(self.varinfo, self.longitude),
                expected_override_dimensions_AM,
            )

        with self.subTest(
            'Retrieves expected override dimensions for the latitude variable'
        ):
            self.assertListEqual(
                get_dimension_array_names(self.varinfo, self.latitude),
                expected_override_dimensions_AM,
            )

        with self.subTest(
            'Retrieves expected override dimensions science variable with a different grid'
        ):
            self.assertListEqual(
                get_dimension_array_names(
                    self.varinfo, '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm'
                ),
                expected_override_dimensions_PM,
            )
        with self.subTest(
            'Retrieves empty dimensions list when science variable has no coordinates'
        ):
            self.assertListEqual(
                get_dimension_array_names(
                    self.varinfo, '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm'
                ),
                expected_override_dimensions_PM,
            )
        with self.subTest(
            'Retrieves expected dimension names for the 3D json configured dimensions'
        ):
            dimension_names_3d = get_dimension_array_names(
                self.smap_ftp_varinfo, '/Freeze_Thaw_Retrieval_Data_Global/surface_flag'
            )
            self.assertEqual(
                dimension_names_3d,
                [
                    '/Freeze_Thaw_Retrieval_Data_Global/am_pm',
                    '/Freeze_Thaw_Retrieval_Data_Global/y_dim',
                    '/Freeze_Thaw_Retrieval_Data_Global/x_dim',
                ],
            )

    def test_get_row_col_sizes_from_coordinates(self):
        """Ensure that the correct row and column sizes are
        returned for the requested coordinates
        """

        with self.subTest('Retrieves the expected row col sizes from the coordinates'):
            expected_row_col_sizes = (5, 10)
            self.assertEqual(
                get_row_col_sizes_from_coordinates(
                    self.lat_arr, self.lon_arr, dim_order_is_y_x=True
                ),
                expected_row_col_sizes,
            )
        with self.subTest('Retrieves the expected row col sizes for the dim array'):
            self.assertEqual(
                get_row_col_sizes_from_coordinates(
                    np.array([1, 2, 3, 4]),
                    np.array([5, 6, 7, 8, 9]),
                    dim_order_is_y_x=True,
                ),
                (4, 5),
            )
        with self.subTest('Retrieves the expected row col sizes for the dim array'):
            self.assertEqual(
                get_row_col_sizes_from_coordinates(
                    np.array([1, 2, 3, 4]),
                    np.array([5, 6, 7, 8, 9]),
                    dim_order_is_y_x=False,
                ),
                (5, 4),
            )
        with self.subTest(
            'Raises an exception when the lat and lon array shapes do not match'
        ):
            lat_mismatched_array = np.array([[1, 2, 3], [3, 4, 5]])
            lon_mismatched_array = np.array([[6, 7], [8, 9], [10, 11]])
            with self.assertRaises(IncompatibleCoordinateVariables) as context:
                get_row_col_sizes_from_coordinates(
                    lat_mismatched_array, lon_mismatched_array, dim_order_is_y_x=True
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
                get_row_col_sizes_from_coordinates(
                    lat_empty_size_array, np.array([5, 6, 7, 8]), dim_order_is_y_x=True
                )

        with self.subTest(
            'Raises an exception when Both arrays are 1-D, but longitude has a zero size'
        ):
            lon_empty_size_array = np.array([])
            with self.assertRaises(IncompatibleCoordinateVariables) as context:
                get_row_col_sizes_from_coordinates(
                    np.array([6, 7, 8, 9]), lon_empty_size_array, dim_order_is_y_x=True
                )

        with self.subTest(
            'Raises an exception when latitude array that is zero dimensional'
        ):
            lat_empty_ndim_array = np.array(
                0,
            )
            with self.assertRaises(IncompatibleCoordinateVariables) as context:
                get_row_col_sizes_from_coordinates(
                    lat_empty_ndim_array, np.array([1, 2, 3, 4]), dim_order_is_y_x=True
                )

        with self.subTest(
            'Raises an exception when longitude array that is zero dimensional'
        ):
            lon_empty_ndim_array = np.array(
                0,
            )
            with self.assertRaises(IncompatibleCoordinateVariables) as context:
                get_row_col_sizes_from_coordinates(
                    np.array([1, 2, 3, 4]), lon_empty_ndim_array, dim_order_is_y_x=True
                )
        with self.subTest('when lat/lon arr is more than 2D'):
            lat_3d_array = np.arange(24).reshape(2, 3, 4)
            lon_3d_array = np.arange(24).reshape(2, 3, 4)
            self.assertEqual(
                get_row_col_sizes_from_coordinates(
                    lat_3d_array, lon_3d_array, dim_order_is_y_x=True
                ),
                (3, 4),
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

    def test_get_max_spread_pts(self):
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
            actual_indices = get_max_spread_pts(~valid_values)
            self.assertTrue(actual_indices[0] == expected_indices[0])
            self.assertTrue(actual_indices[1] == expected_indices[1])

        with self.subTest('With just one valid index in the coordinates'):
            valid_values = np.array(
                [
                    [False, False, False],
                    [False, True, False],
                    [False, False, False],
                ]
            )
            with self.assertRaises(InvalidCoordinateData) as context:
                get_max_spread_pts(~valid_values)
                self.assertEqual(
                    context.exception.message,
                    'Only one valid point in coordinate data',
                )

        with self.subTest('No valid points from coordinates'):
            valid_values = np.array(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ]
            )
            with self.assertRaises(InvalidCoordinateData) as context:
                get_max_spread_pts(~valid_values)
                self.assertEqual(
                    context.exception.message,
                    'No valid coordinate data',
                )

    def test_get_valid_sample_pts(self):
        """Ensure that two sets of valid indices are
        returned by the method with a set of lat/lon coordinates as input

        """
        with self.subTest('Get two sets of valid indices from coordinates dataset'):
            expected_grid_indices = (
                [[0, 0], [4, 0]],
                [[0, 0], [0, 9]],
            )
            actual_row_indices, actual_col_indices = get_valid_sample_pts(
                self.lat_arr,
                self.lon_arr,
                self.varinfo.get_variable(self.latitude),
                self.varinfo.get_variable(self.longitude),
            )
            self.assertListEqual(actual_row_indices, expected_grid_indices[0])
            self.assertListEqual(actual_col_indices, expected_grid_indices[1])
            self.assertTupleEqual(
                (actual_row_indices, actual_col_indices), expected_grid_indices
            )
        with self.subTest('Only a single valid point in coordinates dataset'):
            lat_arr = np.array(
                [
                    [-9999.0, -9999.0, 40.1, -9999.0, -9999.0],
                    [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0],
                ]
            )
            lon_arr = np.array(
                [
                    [-9999.0, -9999.0, 100.1, -9999.0, -9999.0],
                    [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0],
                ]
            )
            with self.assertRaises(InvalidCoordinateData) as context:
                get_valid_sample_pts(
                    lat_arr,
                    lon_arr,
                    self.varinfo.get_variable(self.latitude),
                    self.varinfo.get_variable(self.longitude),
                )
                self.assertEqual(
                    context.exception.message,
                    'No valid coordinate data',
                )
        with self.subTest('valid points in one row in coordinates dataset'):
            lat_arr = np.array(
                [
                    [40.1, 40.1, 40.1, 40.1, 40.1],
                    [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0],
                ]
            )
            lon_arr = np.array(
                [
                    [-179.0, -10.0, 100.1, 130.0, 179.0],
                    [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0],
                ]
            )
            with self.assertRaises(InvalidCoordinateData) as context:
                get_valid_sample_pts(
                    lat_arr,
                    lon_arr,
                    self.varinfo.get_variable(self.latitude),
                    self.varinfo.get_variable(self.longitude),
                )
                self.assertEqual(
                    context.exception.message,
                    'No valid coordinate data',
                )
        with self.subTest('valid points in one column in coordinates dataset'):
            lat_arr = np.array(
                [
                    [-9999.0, -9999.0, 40.1, -9999.0, -9999.0],
                    [-9999.0, -9999.0, -50.0, -9999.0, -9999.0],
                ]
            )
            lon_arr = np.array(
                [
                    [-9999.0, -9999.0, 100.1, -9999.0, -9999.0],
                    [-9999.0, -9999.0, 100.1, -9999.0, -9999.0],
                ]
            )
            with self.assertRaises(InvalidCoordinateData) as context:
                get_valid_sample_pts(
                    lat_arr,
                    lon_arr,
                    self.varinfo.get_variable(self.latitude),
                    self.varinfo.get_variable(self.longitude),
                )
                self.assertEqual(
                    context.exception.message,
                    'No valid coordinate data',
                )
        with self.subTest('no valid points in coordinates dataset'):
            lat_arr = np.array(
                [
                    [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0],
                    [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0],
                ]
            )
            lon_arr = np.array(
                [
                    [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0],
                    [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0],
                ]
            )
            with self.assertRaises(InvalidCoordinateData) as context:
                get_valid_sample_pts(
                    lat_arr,
                    lon_arr,
                    self.varinfo.get_variable(self.latitude),
                    self.varinfo.get_variable(self.longitude),
                )
                self.assertEqual(
                    context.exception.message,
                    'No valid coordinate data',
                )

    def test_get_dimension_order_and_dim_values(self):
        """Ensure that the correct dimension order index
        is returned with a geo array of lat/lon values
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
        with self.subTest('Get y_x order when projected_dim is changing across row'):
            row_indices = [[0, 0], [4, 0]]
            expected_dim_values = [7341677.255608977, -7338157.219843731]
            y_x_order, dim_values = get_dimension_order_and_dim_values(
                self.lat_arr, self.lon_arr, row_indices, crs, is_row=True
            )
            self.assertEqual(y_x_order, True)
            self.assertListEqual(dim_values, expected_dim_values)
        with self.subTest('Get y_x order when projected_dim is changing across column'):
            col_indices = [[0, 0], [0, 9]]
            expected_dim_values = [-17299990.048985746, 17213152.396759935]
            y_x_order, dim_values = get_dimension_order_and_dim_values(
                self.lat_arr, self.lon_arr, col_indices, crs, is_row=False
            )
            self.assertEqual(y_x_order, True)
            self.assertListEqual(dim_values, expected_dim_values)
        with self.subTest('Get x_y order when projected_dim is changing across row'):
            row_indices = [[0, 0], [4, 0]]
            expected_dim_values = [-17299990.048985746, 17213152.396759935]
            y_x_order, dim_values = get_dimension_order_and_dim_values(
                self.lat_arr_reversed,
                self.lon_arr_reversed,
                row_indices,
                crs,
                is_row=True,
            )
            self.assertEqual(y_x_order, False)
            self.assertListEqual(dim_values, expected_dim_values)
        with self.subTest('Get x_y order when projected_dim is changing across column'):
            col_indices = [[0, 0], [0, 9]]
            expected_dim_values = [7341677.255608977, -7341677.255608977]
            y_x_order, dim_values = get_dimension_order_and_dim_values(
                self.lat_arr_reversed,
                self.lon_arr_reversed,
                col_indices,
                crs,
                is_row=False,
            )
            self.assertEqual(y_x_order, False)
            self.assertListEqual(dim_values, expected_dim_values)
        with self.subTest('Get y_x order when projected_dims are not varying'):
            lat_arr = np.array(
                [
                    [89.1, 89.1, 89.1],
                    [89.1, 89.1, 89.1],
                    [89.1, 89.1, 89.1],
                    [89.1, 89.1, 89.1],
                    [89.1, 89.1, 89.1],
                ]
            )
            lon_arr = np.array(
                [
                    [-178.1, -178.1, -178.1],
                    [-178.1, -178.1, -178.1],
                    [-178.1, -178.1, -178.1],
                    [-178.1, -178.1, -178.1],
                    [-178.1, -178.1, -178.1],
                ]
            )
            row_indices = [[0, 0], [4, 0]]
            with self.assertRaises(InvalidCoordinateData) as context:
                get_dimension_order_and_dim_values(
                    lat_arr, lon_arr, row_indices, crs, is_row=True
                )
                self.assertEqual(
                    context.exception.message,
                    'lat/lon values are constant',
                )
        with self.subTest(
            'Get y_x order with 3 dimensions and values changing across row'
        ):
            row_indices = [[0, 0], [4, 0]]
            expected_dim_values = [7341677.255608977, -7338157.219843731]
            y_x_order, dim_values = get_dimension_order_and_dim_values(
                self.lat_arr_3d, self.lon_arr_3d, row_indices, crs, is_row=True
            )
            self.assertEqual(y_x_order, True)
            self.assertListEqual(dim_values, expected_dim_values)

    def test_create_dimension_arrays_from_coordinates(
        self,
    ):
        """Ensure that the correct x and y dim arrays
        are returned from a lat/lon prefetch dataset and
        crs provided.
        """
        smap_varinfo = VarInfoFromDmr(
            'tests/data/SC_SPL3SMP_008.dmr',
            'SPL3SMP',
            'hoss/hoss_config.json',
        )
        smap_file_path = 'tests/data/SC_SPL3SMP_008_prefetch.nc4'

        latitude_coordinate = smap_varinfo.get_variable(
            '/Soil_Moisture_Retrieval_Data_AM/latitude'
        )
        longitude_coordinate = smap_varinfo.get_variable(
            '/Soil_Moisture_Retrieval_Data_AM/longitude'
        )
        projected_dimension_names_am = [
            '/Soil_Moisture_Retrieval_Data_AM/dim_y',
            '/Soil_Moisture_Retrieval_Data_AM/dim_x',
        ]
        projected_dimension_names_pm = [
            '/Soil_Moisture_Retrieval_Data_PM/dim_y',
            '/Soil_Moisture_Retrieval_Data_PM/dim_x',
        ]
        crs = CRS.from_cf(
            {
                'false_easting': 0.0,
                'false_northing': 0.0,
                'longitude_of_central_meridian': 0.0,
                'standard_parallel': 30.0,
                'grid_mapping_name': 'lambert_cylindrical_equal_area',
            }
        )
        expected_xdim = np.array([-17349514.353068016, 17349514.353068016])
        expected_ydim = np.array([7296524.6913595535, -7296524.691359556])

        with self.subTest('Projected x-y dim arrays from coordinate datasets'):
            with Dataset(smap_file_path, 'r') as smap_prefetch:
                x_y_dim_am = create_dimension_arrays_from_coordinates(
                    smap_prefetch,
                    latitude_coordinate,
                    longitude_coordinate,
                    crs,
                    projected_dimension_names_am,
                )
                x_y_dim_pm = create_dimension_arrays_from_coordinates(
                    smap_prefetch,
                    latitude_coordinate,
                    longitude_coordinate,
                    crs,
                    projected_dimension_names_pm,
                )

                self.assertListEqual(
                    list(x_y_dim_am.keys()), projected_dimension_names_am
                )
                self.assertListEqual(
                    list(x_y_dim_pm.keys()), projected_dimension_names_pm
                )
                self.assertEqual(
                    x_y_dim_am['/Soil_Moisture_Retrieval_Data_AM/dim_y'][0],
                    expected_ydim[0],
                )
                self.assertEqual(
                    x_y_dim_am['/Soil_Moisture_Retrieval_Data_AM/dim_y'][-1],
                    expected_ydim[-1],
                )
                self.assertEqual(
                    x_y_dim_am['/Soil_Moisture_Retrieval_Data_AM/dim_x'][0],
                    expected_xdim[0],
                )
                self.assertEqual(
                    x_y_dim_am['/Soil_Moisture_Retrieval_Data_AM/dim_x'][-1],
                    expected_xdim[-1],
                )
                self.assertEqual(
                    x_y_dim_pm['/Soil_Moisture_Retrieval_Data_PM/dim_y'][0],
                    expected_ydim[0],
                )
                self.assertEqual(
                    x_y_dim_pm['/Soil_Moisture_Retrieval_Data_PM/dim_y'][-1],
                    expected_ydim[-1],
                )
                self.assertEqual(
                    x_y_dim_pm['/Soil_Moisture_Retrieval_Data_PM/dim_x'][0],
                    expected_xdim[0],
                )
                self.assertEqual(
                    x_y_dim_pm['/Soil_Moisture_Retrieval_Data_PM/dim_x'][-1],
                    expected_xdim[-1],
                )
        with self.subTest('Invalid data in coordinate datasets'):
            prefetch = {
                '/Soil_Moisture_Retrieval_Data_AM/latitude': np.array(
                    [
                        [89.3, 89.3, -9999, 89.3, 89.3],
                        [-9999, -9999, -60.2, -60.2, -60.2],
                        [-88.1, -9999, -88.1, -88.1, -88.1],
                    ]
                ),
                '/Soil_Moisture_Retrieval_Data_AM/longitude': np.array(
                    [
                        [-9999, -9999, -9999, -9999, 178.4],
                        [-179.3, -9999, -9999, -9999, -9999],
                        [-179.3, -9999, -9999, -9999, -9999],
                    ]
                ),
            }
            with self.assertRaises(InvalidCoordinateData) as context:
                create_dimension_arrays_from_coordinates(
                    prefetch,
                    latitude_coordinate,
                    longitude_coordinate,
                    crs,
                    projected_dimension_names_am,
                )
                self.assertEqual(
                    context.exception.message,
                    'lat/lon values are constant',
                )
        with self.subTest('Cannot determine x-y order in coordinate datasets'):
            prefetch = {
                '/Soil_Moisture_Retrieval_Data_AM/latitude': np.array(
                    [
                        [89.3, 89.3, -9999, 89.3, 89.3],
                        [-9999, -9999, 89.3, 89.3, 89.3],
                        [89.3, 89.3, 89.3, 89.3, 89.3],
                    ]
                ),
                '/Soil_Moisture_Retrieval_Data_AM/longitude': np.array(
                    [
                        [-9999, -9999, -9999, -9999, 178.4],
                        [-179.3, -9999, -9999, -9999, -9999],
                        [-179.3, -9999, -9999, -9999, -9999],
                    ]
                ),
            }
            with self.assertRaises(InvalidCoordinateData) as context:
                create_dimension_arrays_from_coordinates(
                    prefetch,
                    latitude_coordinate,
                    longitude_coordinate,
                    crs,
                    projected_dimension_names_am,
                )
                self.assertEqual(
                    context.exception.message,
                    'lat/lon values are constant',
                )
        with self.subTest('Projected x-y dim arrays from coordinate datasets'):
            prefetch_x_y = {
                '/Soil_Moisture_Retrieval_Data_AM/latitude': self.lat_arr_reversed,
                '/Soil_Moisture_Retrieval_Data_AM/longitude': self.lon_arr_reversed,
            }
            with self.assertRaises(UnsupportedDimensionOrder) as context:
                x_y_dim_am = create_dimension_arrays_from_coordinates(
                    prefetch_x_y,
                    latitude_coordinate,
                    longitude_coordinate,
                    crs,
                    projected_dimension_names_am,
                )

    def test_create_dimension_arrays_from_3d_coordinates(
        self,
    ):
        """Ensure that the correct x and y dim arrays
        are returned from a lat/lon prefetch dataset and
        crs provided.
        """

        latitude_coordinate = self.smap_ftp_varinfo.get_variable(
            '/Freeze_Thaw_Retrieval_Data_Global/latitude'
        )
        longitude_coordinate = self.smap_ftp_varinfo.get_variable(
            '/Freeze_Thaw_Retrieval_Data_Global/longitude'
        )
        dimension_names_global = [
            '/Freeze_Thaw_Retrieval_Data_Global/am_pm',
            '/Freeze_Thaw_Retrieval_Data_Global/y_dim',
            '/Freeze_Thaw_Retrieval_Data_Global/x_dim',
        ]

        crs = CRS.from_cf(
            {
                'false_easting': 0.0,
                'false_northing': 0.0,
                'longitude_of_central_meridian': 0.0,
                'standard_parallel': 30.0,
                'grid_mapping_name': 'lambert_cylindrical_equal_area',
            }
        )
        expected_xdim = np.array([-17349514.353068016, 17349514.353068016])
        expected_ydim = np.array([7296524.6913595535, -7296509.222123815])

        with self.subTest('Projected x-y dim arrays from coordinate datasets'):
            with Dataset(self.smap_ftp_file_path, 'r') as smap_prefetch:
                x_y_dim_global = create_dimension_arrays_from_coordinates(
                    smap_prefetch,
                    latitude_coordinate,
                    longitude_coordinate,
                    crs,
                    dimension_names_global,
                )

                self.assertListEqual(
                    list(x_y_dim_global.keys()),
                    [dimension_names_global[1], dimension_names_global[2]],
                )
                self.assertEqual(
                    x_y_dim_global['/Freeze_Thaw_Retrieval_Data_Global/y_dim'][0],
                    expected_ydim[0],
                )
                self.assertEqual(
                    x_y_dim_global['/Freeze_Thaw_Retrieval_Data_Global/y_dim'][-1],
                    expected_ydim[-1],
                )
                self.assertEqual(
                    x_y_dim_global['/Freeze_Thaw_Retrieval_Data_Global/x_dim'][0],
                    expected_xdim[0],
                )
                self.assertEqual(
                    x_y_dim_global['/Freeze_Thaw_Retrieval_Data_Global/x_dim'][-1],
                    expected_xdim[-1],
                )
