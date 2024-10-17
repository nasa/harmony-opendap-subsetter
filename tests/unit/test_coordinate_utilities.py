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
    get_coordinate_variables,
    get_dimension_scale_from_dimvalues,
    get_fill_values_for_coordinates,
    get_lat_lon_arrays,
    get_override_projected_dimension_name,
    get_override_projected_dimensions,
    get_row_col_sizes_from_coordinate_datasets,
    get_two_valid_geo_grid_points,
    get_valid_indices,
    get_valid_indices_in_dataset,
    get_variables_with_anonymous_dims,
    get_x_y_values_from_geographic_points,
    update_dimension_variables,
)
from hoss.exceptions import (
    CannotComputeDimensionResolution,
    InvalidCoordinateVariable,
    MissingCoordinateVariable,
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

    def get_coordinate_variables(self):
        """Ensure that the correct coordinate variables are
        retrieved for the reqquested science variable

        """

        requested_science_variables = set(
            '/Soil_Moisture_Retrieval_Data_AM/surface_flag',
            '/Soil_Moisture_Retrieval_Data_AM/landcover_class',
            '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm',
        )

        expected_coordinate_variables = tuple([self.latitude], [self.longitude])

        with self.subTest('Retrieves expected coordinates for the requested variable'):
            self.assertTupleEqual(
                get_coordinate_variables(self.varinfo, requested_science_variables),
                expected_coordinate_variables,
            )

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
            with self.assertRaises(CannotComputeDimensionResolution) as context:
                get_dimension_scale_from_dimvalues(
                    dim_values_invalid, dim_indices_asc, dim_size_asc
                )
            self.assertEqual(
                context.exception.message,
                'Cannot compute the dimension resolution for '
                'dim_value: "2" dim_index: "0"',
            )
        with self.subTest('invalid dimension indices'):
            with self.assertRaises(CannotComputeDimensionResolution) as context:
                get_dimension_scale_from_dimvalues(
                    dim_values_desc, dim_indices_invalid, dim_size_desc
                )
            self.assertEqual(
                context.exception.message,
                'Cannot compute the dimension resolution for '
                'dim_value: "100" dim_index: "5"',
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

    def test_get_two_valid_geo_grid_points(self):
        """Ensure that two valid lat/lon points returned by the method
        with a set of lat/lon coordinates as input

        """
        lat_fill = -9999.0
        lon_fill = -9999.0
        row_size = 406
        col_size = 964

        expected_geo_grid_points = [(-179.3, 89.3), (-120.2, -88.1)]
        expected_geo_grid_points_reversed = [(-179.3, 89.3), (178.4, 79.3)]
        with self.subTest('Get two valid geo grid points from coordinates'):
            actual_geo_grid_points = get_two_valid_geo_grid_points(
                self.lat_arr, self.lon_arr, lat_fill, lon_fill, row_size, col_size
            )
            for actual_geo_grid_point, expected_geo_grid_point in zip(
                actual_geo_grid_points.values(), expected_geo_grid_points
            ):
                self.assertEqual(actual_geo_grid_point, expected_geo_grid_point)

        with self.subTest('Get two valid geo grid points from reversed coordinates'):
            actual_geo_grid_points = get_two_valid_geo_grid_points(
                self.lat_arr_reversed,
                self.lon_arr_reversed,
                lat_fill,
                lon_fill,
                row_size,
                col_size,
            )
            for actual_geo_grid_point, expected_geo_grid_point in zip(
                actual_geo_grid_points.values(), expected_geo_grid_points_reversed
            ):
                self.assertEqual(actual_geo_grid_point, expected_geo_grid_point)


# get_dimension_scale_from_dimvalues(
#     dim_values: ndarray, dim_indices: ndarray, dim_size: float
# ) -> ndarray:
