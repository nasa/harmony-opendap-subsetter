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
    get_geo_grid_corners,
    get_lat_lon_arrays,
    get_override_projected_dimension_name,
    get_override_projected_dimensions,
    get_row_col_sizes_from_coordinate_datasets,
    get_valid_indices,
    get_variables_with_anonymous_dims,
    get_x_y_extents_from_geographic_points,
    is_lat_lon_ascending,
    update_dimension_variables,
)
from hoss.exceptions import MissingCoordinateDataset


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

    def test_is_lat_lon_ascending(self):
        """Ensure that latitude and longitude values are correctly identified as
        ascending or descending.

        """

        expected_result = False, True
        with self.subTest('ascending order even with fill values'):
            self.assertEqual(
                is_lat_lon_ascending(self.lat_arr, self.lon_arr, -9999, -9999),
                expected_result,
            )

    def test_get_geo_grid_corners(self):
        """Ensure that the correct corner points returned by the methos
        with a set of lat/lon coordinates as input

        """
        prefetch_dataset = Dataset(self.nc4file, 'r+')
        lat_fill = -9999.0
        lon_fill = -9999.0

        # lat_arr = prefetch_dataset[self.latitude][:]
        # lon_arr = prefetch_dataset[self.longitude][:]

        expected_geo_corners = [
            (-179.3, 89.3),
            (178.4, 89.3),
            (178.4, -88.1),
            (-179.3, -88.1),
        ]

        with self.subTest('Get geo grid corners from coordinates'):
            actual_geo_corners = get_geo_grid_corners(
                self.lat_arr,
                self.lon_arr,
                lat_fill,
                lon_fill,
            )
            for actual, expected in zip(actual_geo_corners, expected_geo_corners):
                self.assertAlmostEqual(actual[0], expected[0], places=1)
                self.assertAlmostEqual(actual[1], expected[1], places=1)

        prefetch_dataset.close()
