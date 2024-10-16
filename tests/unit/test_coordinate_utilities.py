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
    get_fill_values_for_coordinates,
    get_lat_lon_arrays,
    get_override_projected_dimension_name,
    get_override_projected_dimensions,
    get_row_col_sizes_from_coordinate_datasets,
    get_two_valid_geo_grid_points,
    get_valid_indices,
    get_variables_with_anonymous_dims,
    update_dimension_variables,
)
from hoss.exceptions import InvalidCoordinateVariable, MissingCoordinateVariable


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

    def setUp(self):
        """Create fixtures that should be unique per test."""
        self.temp_dir = mkdtemp()

    def tearDown(self):
        """Remove per-test fixtures."""
        if exists(self.temp_dir):
            rmtree(self.temp_dir)

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
        prefetch_dataset = Dataset(self.nc4file, 'r')
        lat_fill = -9999.0
        lon_fill = -9999.0
        row_size = 406
        col_size = 964

        expected_geo_grid_points = [(-179.3, 89.3), (-120.2, -88.1)]

        with self.subTest('Get two valid geo grid points from coordinates'):
            actual_geo_grid_points = get_two_valid_geo_grid_points(
                self.lat_arr, self.lon_arr, lat_fill, lon_fill, row_size, col_size
            )
            for actual, expected in zip(
                actual_geo_grid_points.values(), expected_geo_grid_points
            ):
                self.assertEqual(actual[0], expected[0])
                self.assertEqual(actual[1], expected[1])

        prefetch_dataset.close()
