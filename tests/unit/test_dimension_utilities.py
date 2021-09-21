from logging import getLogger
from unittest import TestCase
from unittest.mock import patch

from harmony.util import config
from varinfo import VarInfoFromDmr
import numpy as np

from pymods.dimension_utilities import (add_index_range,
                                        get_dimension_index_range,
                                        get_fill_slice, is_dimension_ascending,
                                        prefetch_dimension_variables)


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
            grid-dimension variables.

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
        spatial_dimensions = {'/latitude', '/longitude'}

        self.assertEqual(prefetch_dimension_variables(url, self.varinfo,
                                                      required_variables,
                                                      output_dir,
                                                      self.logger,
                                                      access_token,
                                                      self.config),
                         prefetch_path)

        mock_get_opendap_nc4.assert_called_once_with(url, spatial_dimensions,
                                                     output_dir, self.logger,
                                                     access_token, self.config)
