from logging import getLogger
from unittest import TestCase
from unittest.mock import patch

from harmony.util import config
from harmony.message import Dimension
from varinfo import VarInfoFromDmr
import numpy as np

from pymods.bbox_utilities import BBox
from pymods.dimension_utilities import (add_index_range, get_dimension_extents,
                                        get_dimension_index_range,
                                        get_fill_slice,
                                        get_requested_index_ranges,
                                        is_dimension_ascending,
                                        is_index_subset,
                                        prefetch_dimension_variables)
from pymods.exceptions import InvalidNamedDimension


class TestDimensionUtilities(TestCase):
    """ A class for testing functions in the `pymods.dimension_utilities`
        module.

    """
    @classmethod
    def setUpClass(cls):
        cls.config = config(validate=False)
        cls.logger = getLogger('tests')
        cls.varinfo = VarInfoFromDmr('tests/data/rssmif16d_example.dmr',
                                     cls.logger,
                                     'tests/data/test_subsetter_config.yml')

    def test_is_dimension_ascending(self):
        """ Ensure that a dimension variable is correctly identified as
            ascending or descending. This should be immune to having a few
            fill values, particularly in the first and last element in the
            array.

        """
        ascending_data = np.linspace(0, 200, 101)
        descending_data = np.linspace(200, 0, 101)

        # Create a mask that will mask the first and last element of an array
        mask = np.zeros(ascending_data.shape)
        mask[0] = 1
        mask[-1] = 1

        ascending_dimension = np.ma.masked_array(data=ascending_data)
        descending_dimension = np.ma.masked_array(data=descending_data)
        ascending_masked = np.ma.masked_array(data=ascending_data, mask=mask)
        descending_masked = np.ma.masked_array(data=descending_data, mask=mask)

        test_args = [
            ['Ascending dimension returns True', ascending_dimension, True],
            ['Ascending masked dimension returns True', ascending_masked, True],
            ['Descending dimension returns False', descending_dimension, False],
            ['Descending masked dimension returns False', descending_masked, False]
        ]
        for description, dimension, expected_result in test_args:
            with self.subTest(description):
                self.assertEqual(is_dimension_ascending(dimension),
                                 expected_result)

    def test_get_dimension_index_range(self):
        """ Ensure the expected index values are retrieved for the minimum and
            maximum values of an expected range. This should correspond to the
            nearest integer, to ensure partial pixels are included in a
            bounding box spatial subset. List elements must be integers for
            later array slicing.

            data_ascending[20] = data_descending[80] = 40.0
            data_ascending[87] = data_descending[13] = 174.0

            This test should also ensure that extent values exactly halfway
            between pixels should not include the outer pixel.

        """
        data_ascending = np.linspace(0, 200, 101)
        data_descending = np.linspace(200, 0, 101)

        test_args = [
            ['Ascending dimension', data_ascending, 39, 174.3, (20, 87)],
            ['Descending dimension', data_descending, 174.3, 39, (13, 80)],
            ['Ascending halfway between', data_ascending, 39, 175, (20, 87)],
            ['Descending halfway between', data_descending, 175, 39, (13, 80)],
            ['Single point inside pixel', data_ascending, 10, 10, (5, 5)],
            ['Single point on pixel edges', data_ascending, 9, 9, (4, 5)],
        ]

        for description, data, min_extent, max_extent, expected_results in test_args:
            with self.subTest(description):
                dimension_data = np.ma.masked_array(data=data, mask=False)
                results = get_dimension_index_range(dimension_data,
                                                    min_extent,
                                                    max_extent)

                self.assertIsInstance(results[0], int)
                self.assertIsInstance(results[1], int)
                self.assertTupleEqual(results, expected_results)

    def test_add_index_range(self):
        """ Ensure the correct combinations of index ranges are added as
            suffixes to the input variable based upon that variable's
            dimensions.

            If a dimension range has the lower index > upper index, that
            indicates the bounding box crosses the edge of the grid. In this
            instance, the full range of the variable should be retrieved.

            The order of indices in RSSMIF16D is: (time, latitude, longitude)

        """
        with self.subTest('No index constraints'):
            index_ranges = {}
            self.assertEqual(add_index_range('/sst_dtime', self.varinfo,
                                             index_ranges),
                             '/sst_dtime')

        with self.subTest('With index constraints'):
            index_ranges = {'/latitude': [12, 34], '/longitude': [45, 56]}
            self.assertEqual(add_index_range('/sst_dtime', self.varinfo,
                                             index_ranges),
                             '/sst_dtime[][12:34][45:56]')

        with self.subTest('With a longitude crossing discontinuity'):
            index_ranges = {'/latitude': [12, 34], '/longitude': [56, 5]}
            self.assertEqual(add_index_range('/sst_dtime', self.varinfo,
                                             index_ranges),
                             '/sst_dtime[][12:34][]')

    def test_get_fill_slice(self):
        """ Ensure that a slice object is correctly formed for a requested
            dimension.

        """
        fill_ranges = {'/longitude': [200, 15]}

        with self.subTest('An unfilled dimension returns slice(None).'):
            self.assertEqual(
                get_fill_slice('/time', fill_ranges),
                slice(None)
            )

        with self.subTest('A filled dimension returns slice(start, stop).'):
            self.assertEqual(
                get_fill_slice('/longitude', fill_ranges),
                slice(16, 200)
            )

    @patch('pymods.dimension_utilities.get_opendap_nc4')
    def test_prefetch_dimension_variables(self, mock_get_opendap_nc4):
        """ Ensure that when a list of required variables is specified, a
            request to OPeNDAP will be sent requesting only those that are
            grid-dimension variables (both spatial and temporal).

            At this point only spatial dimensions will be included in a
            prefetch request.

        """
        prefetch_path = 'prefetch.nc4'
        mock_get_opendap_nc4.return_value = prefetch_path

        access_token = 'access'
        output_dir = 'tests/output'
        url = 'https://url_to_opendap_granule'
        required_variables = {'/latitude', '/longitude', '/time',
                              '/wind_speed'}
        required_dimensions = {'/latitude', '/longitude', '/time'}

        self.assertEqual(prefetch_dimension_variables(url, self.varinfo,
                                                      required_variables,
                                                      output_dir,
                                                      self.logger,
                                                      access_token,
                                                      self.config),
                         prefetch_path)

        mock_get_opendap_nc4.assert_called_once_with(url, required_dimensions,
                                                     output_dir, self.logger,
                                                     access_token, self.config)

    def test_get_dimension_extents(self):
        """ Ensure that the expected dimension extents are retrieved.

            The three grids below correspond to longitude dimensions of three
            collections used with HOSS:

            * GPM: -180 ≤ longitude (degrees east) ≤ 180.
            * RSSMIF16D: 0 ≤ longitude (degrees east) ≤ 360.
            * MERRA-2: -180.3125 ≤ longitude (degrees east) ≤ 179.6875.

            These represent fully wrapped longitudes (GPM), fully unwrapped
            longitudes (RSSMIF16D) and partially wrapped longitudes (MERRA-2).

        """
        gpm_lons = np.linspace(-179.950, 179.950, 3600)
        rss_lons = np.linspace(0.125, 359.875, 1440)
        merra_lons = np.linspace(-180.0, 179.375, 576)

        test_args = [
            ['Fully wrapped dimension', gpm_lons, -180, 180],
            ['Fully unwrapped dimension', rss_lons, 0, 360],
            ['Partially wrapped dimension', merra_lons, -180.3125, 179.6875]
        ]

        for description, dim_array, expected_min, expected_max in test_args:
            with self.subTest(description):
                np.testing.assert_almost_equal(
                    get_dimension_extents(dim_array),
                    (expected_min, expected_max)
                )

    def test_is_index_subset(self):
        """ Ensure the function correctly determines when a HOSS request will
            be an index subset (i.e., bounding box, shape file or temporal).

        """
        bounding_box = BBox(10, 20, 30, 40)
        shape_file_path = 'path/to/shape.geo.json'
        temporal_range = ['2021-01-01T01:30:00', '2021-01-01T02:00:00']
        dim_range = [['lev',800,900]]

        test_args = [
            ['Bounding box', bounding_box, None, None, None],
            ['Shape file', None, shape_file_path, None, None],
            ['Temporal', None, None, None, temporal_range],
            ['Bounding box and temporal', bounding_box, None, None, \
             temporal_range],
            ['Shape file and temporal', None, shape_file_path, None, \
             temporal_range],
            ['Dimension', None, None, dim_range, None],
            ['Bounding box and dimension', bounding_box, None, dim_range, None],
        ]

        for description, bbox, shape_file, dim_request, time_range in test_args:
            with self.subTest(description):
                self.assertTrue(is_index_subset(bbox, shape_file,
                                                dim_request, time_range))

        with self.subTest('Not an index range subset'):
            self.assertFalse(is_index_subset(None, None, None, None))

    def test_get_requested_index_ranges(self):
        """ Ensure the function correctly retrieves all index ranges from
            explicitly named dimensions.

            This test will use the `latitude` and `longitude` variables in the
            RSSMIF16D example files.

            If one extent is not specified, the returned index range should
            extend to either the first or last element (depending on whether
            the omitted extent is a maximum or a minimum and whether the
            dimension array is ascending or descending).

            f16_ssmis_lat_lon_desc.nc has a descending latitude dimension
            array.

        """
        ascending_file = 'tests/data/f16_ssmis_lat_lon.nc'
        descending_file = 'tests/data/f16_ssmis_lat_lon_desc.nc'

        required_variables = {'/rainfall_rate', '/latitude', '/longitude'}

        with self.subTest('Ascending dimension'):
            # 20.0 ≤ latitude[440] ≤ 20.25, 29.75 ≤ latitude[479] ≤ 30.0
            dim_request = [Dimension({'name': '/latitude', 'min': 20, 'max': 30})]
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           ascending_file, dim_request),
                {'/latitude': (440, 479)}
            )

        with self.subTest('Multiple ascending dimensions'):
            # 20.0 ≤ latitude[440] ≤ 20.25, 29.75 ≤ latitude[479] ≤ 30.0
            # 140.0 ≤ longitude[560] ≤ 140.25, 149.75 ≤ longitude[599] ≤ 150.0
            dim_request = [Dimension({'name': '/latitude', 'min': 20, 'max': 30}),
                           Dimension({'name': '/longitude', 'min': 140, 'max': 150})]
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           ascending_file, dim_request),
                {'/latitude': (440, 479), '/longitude': (560, 599)}
            )


        with self.subTest('Descending dimension'):
            # 30.0 ≥ latitude[240] ≥ 29.75, 20.25 ≥ latitude[279] ≥ 20.0
            dim_request = [Dimension({'name': '/latitude', 'min': 20, 'max': 30})]
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           descending_file, dim_request),
                {'/latitude': (240, 279)}
            )

        with self.subTest('Dimension has no leading slash'):
            # 20.0 ≤ latitude[440] ≤ 20.25, 29.75 ≤ latitude[479] ≤ 30.0
            dim_request = [Dimension({'name': 'latitude', 'min': 20, 'max': 30})]
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           ascending_file, dim_request),
                {'/latitude': (440, 479)}
            )

        with self.subTest('Unspecified minimum value'):
            # 29.75 ≤ latitude[479] ≤ 30.0
            dim_request = [Dimension({'name': '/latitude', 'min': None, 'max':30})]
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           ascending_file, dim_request),
                {'/latitude': (0, 479)}
            )

        with self.subTest('Unspecified maximum value'):
            # 20.0 ≤ latitude[440] ≤ 20.25, 179.75 ≤ latitude[719] ≤ 180.0
            dim_request = [Dimension({'name': '/latitude', 'min': 20, 'max': None})]
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           ascending_file, dim_request),
                {'/latitude': (440, 719)}
            )

        with self.subTest('Descending, unspecified minimum value'):
            # 30.0 ≥ latitude[240] ≥ 29.75, 0.25 ≥ latitude[719] ≥ 0.0
            dim_request = [Dimension({'name': '/latitude', 'min': None, 'max': 30})]
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           descending_file, dim_request),
                {'/latitude': (240, 719)}
            )

        with self.subTest('Descending, unspecified maximum value'):
            # 20.25 ≥ latitude[279] ≥ 20.0
            dim_request = [Dimension({'name': '/latitude', 'min': 20, 'max': None})]
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           descending_file, dim_request),
                {'/latitude': (0, 279)}
            )

        with self.subTest('Unrecognised dimension'):
            # Check for a non-existent named dimension
            dim_request = [Dimension({'name': '/FooBar', 'min': None, 'max': 10})]
            with self.assertRaises(InvalidNamedDimension):
                get_requested_index_ranges(required_variables, self.varinfo,
                                           descending_file, dim_request),

