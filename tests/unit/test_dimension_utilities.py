from logging import getLogger
from os.path import exists
from shutil import copy, rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import ANY, patch

from harmony.util import config
from harmony.message import Message
from pathlib import PurePosixPath
from netCDF4 import Dataset
from numpy.ma import masked_array
from numpy.testing import assert_array_equal
from varinfo import VarInfoFromDmr
import numpy as np

from hoss.dimension_utilities import (add_index_range, get_dimension_bounds,
                                      get_dimension_extents,
                                      get_dimension_index_range,
                                      get_dimension_indices_from_bounds,
                                      get_dimension_indices_from_values,
                                      get_fill_slice,
                                      get_requested_index_ranges,
                                      is_almost_in, is_dimension_ascending,
                                      is_index_subset,
                                      prefetch_dimension_variables,
                                      add_bounds_variables,
                                      needs_bounds,
                                      get_bounds_array,
                                      write_bounds)
from hoss.exceptions import InvalidNamedDimension, InvalidRequestedRange


class TestDimensionUtilities(TestCase):
    """ A class for testing functions in the `hoss.dimension_utilities`
        module.

    """
    @classmethod
    def setUpClass(cls):
        """ Create fixtures that can be reused for all tests. """
        cls.config = config(validate=False)
        cls.logger = getLogger('tests')
        cls.varinfo = VarInfoFromDmr(
            'tests/data/rssmif16d_example.dmr',
            config_file='tests/data/test_subsetter_config.json'
        )
        cls.ascending_dimension = masked_array(np.linspace(0, 200, 101))
        cls.descending_dimension = masked_array(np.linspace(200, 0, 101))
        cls.varinfo_with_bounds = VarInfoFromDmr(
            'tests/data/GPM_3IMERGHH_example.dmr'
        )
        cls.bounds_array = np.array([
            [90.0, 89.0], [89.0, 88.0], [88.0, 87.0], [87.0, 86.0],
            [86.0, 85.0], [85.0, 84.0], [84.0, 83.0], [83.0, 82.0],
            [82.0, 81.0], [81.0, 80.0], [80.0, 79.0], [79.0, 78.0],
            [78.0, 77.0], [77.0, 76.0], [76.0, 75.0], [75.0, 74.0],
            [74.0, 73.0], [73.0, 72.0], [72.0, 71.0], [71.0, 70.0],
            [70.0, 69.0], [69.0, 68.0], [68.0, 67.0], [67.0, 66.0],
            [66.0, 65.0], [65.0, 64.0], [64.0, 63.0], [63.0, 62.0],
            [62.0, 61.0], [61.0, 60.0]
        ])

    def setUp(self):
        """ Create fixtures that should be unique per test. """
        self.temp_dir = mkdtemp()

    def tearDown(self):
        """ Remove per-test fixtures. """
        if exists(self.temp_dir):
            rmtree(self.temp_dir)

    def test_is_dimension_ascending(self):
        """ Ensure that a dimension variable is correctly identified as
            ascending or descending. This should be immune to having a few
            fill values, particularly in the first and last element in the
            array.

        """
        # Create a mask that will mask the first and last element of an array
        mask = np.zeros(self.ascending_dimension.shape)
        mask[0] = 1
        mask[-1] = 1

        ascending_masked = masked_array(data=self.ascending_dimension.data,
                                        mask=mask)
        descending_masked = masked_array(data=self.descending_dimension.data,
                                         mask=mask)
        single_element = masked_array(data=np.array([1]))

        test_args = [
            ['Ascending dimension returns True', self.ascending_dimension, True],
            ['Ascending masked dimension returns True', ascending_masked, True],
            ['Single element array returns True', single_element, True],
            ['Descending dimension returns False', self.descending_dimension, False],
            ['Descending masked dimension returns False', descending_masked, False]
        ]
        for description, dimension, expected_result in test_args:
            with self.subTest(description):
                self.assertEqual(is_dimension_ascending(dimension),
                                 expected_result)

    @patch('hoss.dimension_utilities.get_dimension_indices_from_values')
    def test_get_dimension_index_range(self, mock_get_indices_from_values):
        """ Ensure that the dimension variable is correctly determined to be
            ascending or descending, such that `get_dimension_min_max_indices`
            is called with the correct ordering of minimum and maximum values.
            This function should also handle when either the minimum or maximum
            requested value is unspecified, indicating that the beginning or
            end of the array should be used accordingly.

            data_ascending[20] = data_descending[80] = 40.0
            data_ascending[87] = data_descending[13] = 174.0

        """
        requested_min_value = 39.0
        requested_max_value = 174.3

        with self.subTest('Ascending, minimum and maximum extents specified'):
            get_dimension_index_range(self.ascending_dimension,
                                      requested_min_value, requested_max_value)
            mock_get_indices_from_values.called_once_with(
                self.ascending_dimension, requested_min_value,
                requested_max_value
            )
            mock_get_indices_from_values.reset_mock()

        with self.subTest('Ascending, only minimum extent specified'):
            get_dimension_index_range(self.ascending_dimension,
                                      requested_min_value, None)
            mock_get_indices_from_values.called_once_with(
                self.ascending_dimension, requested_min_value,
                self.ascending_dimension[:][-1]
            )
            mock_get_indices_from_values.reset_mock()

        with self.subTest('Ascending, only maximum extent specified'):
            get_dimension_index_range(self.ascending_dimension, None,
                                      requested_max_value)
            mock_get_indices_from_values.called_once_with(
                self.ascending_dimension, self.ascending_dimension[:][0],
                requested_max_value
            )
            mock_get_indices_from_values.reset_mock()

        with self.subTest('Descending, minimum and maximum extents specified'):
            get_dimension_index_range(self.descending_dimension,
                                      requested_min_value, requested_max_value)
            mock_get_indices_from_values.called_once_with(
                self.descending_dimension, requested_max_value,
                requested_min_value
            )
            mock_get_indices_from_values.reset_mock()

        with self.subTest('Descending, only minimum extent specified'):
            get_dimension_index_range(self.descending_dimension,
                                      requested_min_value, None)
            mock_get_indices_from_values.called_once_with(
                self.descending_dimension, self.descending_dimension[:][0],
                requested_min_value
            )
            mock_get_indices_from_values.reset_mock()

        with self.subTest('Descending, only maximum extent specified'):
            get_dimension_index_range(self.descending_dimension, None,
                                      requested_max_value)
            mock_get_indices_from_values.called_once_with(
                self.descending_dimension, requested_max_value,
                self.descending_dimension[:][-1]
            )
            mock_get_indices_from_values.reset_mock()

    @patch('hoss.dimension_utilities.get_dimension_indices_from_values')
    def test_get_dimension_index_range_requested_zero_values(self,
                                                             mock_get_indices_from_values):
        """ Ensure that a 0 is treated correctly, and not interpreted as a
            False boolean value.

        """
        with self.subTest('Ascending dimension values, min = 0'):
            get_dimension_index_range(self.ascending_dimension, 0, 10)
            mock_get_indices_from_values.assert_called_once_with(
                self.ascending_dimension, 0, 10
            )
            mock_get_indices_from_values.reset_mock()

        with self.subTest('Ascending dimension values, max = 0'):
            get_dimension_index_range(self.ascending_dimension, -10, 0)
            mock_get_indices_from_values.assert_called_once_with(
                self.ascending_dimension, -10, 0
            )
            mock_get_indices_from_values.reset_mock()

        with self.subTest('Descending dimension values, min = 0'):
            get_dimension_index_range(self.descending_dimension, 0, 10)
            mock_get_indices_from_values.assert_called_once_with(
                self.descending_dimension, 10, 0
            )
            mock_get_indices_from_values.reset_mock()

        with self.subTest('Descending dimension values, max = 0'):
            get_dimension_index_range(self.descending_dimension, -10, 0)
            mock_get_indices_from_values.assert_called_once_with(
                self.descending_dimension, 0, -10
            )
            mock_get_indices_from_values.reset_mock()

    def test_get_dimension_indices_from_indices(self):
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
        test_args = [
            ['Ascending dimension', self.ascending_dimension, 39, 174.3, (20, 87)],
            ['Descending dimension', self.descending_dimension, 174.3, 39, (13, 80)],
            ['Ascending halfway between', self.ascending_dimension, 39, 175, (20, 87)],
            ['Descending halfway between', self.descending_dimension, 175, 39, (13, 80)],
            ['Single point inside pixel', self.ascending_dimension, 10, 10, (5, 5)],
            ['Single point on pixel edges', self.ascending_dimension, 9, 9, (4, 5)],
        ]

        for description, dimension, min_extent, max_extent, expected_results in test_args:
            with self.subTest(description):
                results = get_dimension_indices_from_values(dimension,
                                                            min_extent,
                                                            max_extent)

                self.assertIsInstance(results[0], int)
                self.assertIsInstance(results[1], int)
                self.assertTupleEqual(results, expected_results)

    def test_add_index_range(self):
        """ Ensure the correct combinations of index ranges are added as
            suffixes to the input variable based upon that variable's dimensions.

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

    @patch('hoss.dimension_utilities.add_bounds_variables')
    @patch('hoss.dimension_utilities.get_opendap_nc4')
    def test_prefetch_dimension_variables(self, mock_get_opendap_nc4,
                                          mock_add_bounds_variables):
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

        mock_add_bounds_variables.assert_called_once_with(prefetch_path,
                                                          required_dimensions,
                                                          self.varinfo, self.logger)

    @patch('hoss.dimension_utilities.needs_bounds')
    @patch('hoss.dimension_utilities.write_bounds')
    def test_add_bounds_variables(self, mock_write_bounds, mock_needs_bounds):
        """ Ensure that `write_bounds` is called when it's needed,
            and that it's not called when it's not needed.

        """
        prefetch_dataset_name = 'tests/data/ATL16_prefetch.nc4'
        varinfo_prefetch = VarInfoFromDmr(
            'tests/data/ATL16_prefetch.dmr'
        )
        required_dimensions = {'/npolar_grid_lat', '/npolar_grid_lon',
                               '/spolar_grid_lat', '/spolar_grid_lon',
                               '/global_grid_lat', '/global_grid_lon'}

        with self.subTest('Bounds need to be written'):
            mock_needs_bounds.return_value = True
            add_bounds_variables(prefetch_dataset_name,
                                 required_dimensions,
                                 varinfo_prefetch,
                                 self.logger)
            self.assertEqual(mock_write_bounds.call_count, 6)

            mock_needs_bounds.reset_mock()
            mock_write_bounds.reset_mock()

        with self.subTest('Bounds should not be written'):
            mock_needs_bounds.return_value = False
            add_bounds_variables(prefetch_dataset_name,
                                 required_dimensions,
                                 varinfo_prefetch,
                                 self.logger)
            mock_write_bounds.assert_not_called()

    def test_needs_bounds(self):
        """ Ensure that the correct boolean value is returned for four
            different cases:

            1) False - cell_alignment[edge] attribute exists and
                       bounds variable already exists.
            2) False - cell_alignment[edge] attribute does not exist and
                       bounds variable already exists.
            3) True  - cell_alignment[edge] attribute exists and
                       bounds variable does not exist.
            4) False - cell_alignment[edge] attribute does not exist and
                       bounds variable does not exist.

        """
        varinfo_bounds = VarInfoFromDmr(
            'tests/data/ATL16_prefetch_bnds.dmr'
        )

        with self.subTest('Variable has cell alignment and bounds'):
            self.assertFalse(needs_bounds(varinfo_bounds.get_variable(
                '/variable_edge_has_bnds')))

        with self.subTest('Variable has no cell alignment and has bounds'):
            self.assertFalse(needs_bounds(varinfo_bounds.get_variable(
                '/variable_no_edge_has_bnds')))

        with self.subTest('Variable has cell alignment and no bounds'):
            self.assertTrue(needs_bounds(varinfo_bounds.get_variable(
                '/variable_edge_no_bnds')))

        with self.subTest('Variable has no cell alignment and no bounds'):
            self.assertFalse(needs_bounds(varinfo_bounds.get_variable(
                '/variable_no_edge_no_bnds')))

    def test_get_bounds_array(self):
        """ Ensure that the expected bounds array is created given
            the input dimension variable values.

        """
        prefetch_dataset = Dataset('tests/data/ATL16_prefetch.nc4', 'r')
        dimension_path = '/npolar_grid_lat'

        expected_bounds_array = self.bounds_array

        assert_array_equal(get_bounds_array(prefetch_dataset,
                                            dimension_path),
                           expected_bounds_array)

    def test_write_bounds(self):
        """ Ensure that bounds data array is written to the dimension
            dataset, both when the dimension variable is in the root group
            and in a nested group.

        """
        varinfo_prefetch = VarInfoFromDmr('tests/data/ATL16_prefetch_group.dmr')
        prefetch_dataset = Dataset('tests/data/ATL16_prefetch_group.nc4', 'r+')

        # Expected variable contents in file.
        expected_bounds_data = self.bounds_array

        with self.subTest('Dimension variable is in the root group'):
            root_variable_full_path = '/npolar_grid_lat'
            root_varinfo_variable = varinfo_prefetch.get_variable(
                root_variable_full_path)
            root_variable_name = 'npolar_grid_lat'
            root_bounds_name = root_variable_name + '_bnds'

            write_bounds(prefetch_dataset, root_varinfo_variable)

            # Check that bounds variable was written to the root group.
            self.assertTrue(prefetch_dataset.variables[root_bounds_name])

            resulting_bounds_root_data = prefetch_dataset.variables[
                root_bounds_name][:]

            assert_array_equal(resulting_bounds_root_data,
                               expected_bounds_data)
            # Check that varinfo variable has 'bounds' attribute.
            self.assertEqual(root_varinfo_variable.attributes['bounds'],
                             root_bounds_name)
            # Check that NetCDF4 dimension variable has 'bounds' attribute.
            self.assertEqual(prefetch_dataset.variables[
                root_variable_name].__dict__.get('bounds'),
                root_bounds_name)
            # Check that VariableFromDmr has 'bounds' reference in
            # the references dictionary.
            self.assertEqual(root_varinfo_variable.references['bounds'],
                             {root_bounds_name, })

        with self.subTest('Dimension variable is in a nested group'):
            nested_variable_full_path = '/group1/group2/zelda'
            nested_varinfo_variable = varinfo_prefetch.get_variable(
                nested_variable_full_path)
            nested_variable_name = 'zelda'
            nested_group_path = '/group1/group2'
            nested_group = prefetch_dataset[nested_group_path]
            nested_bounds_name = nested_variable_name + '_bnds'

            write_bounds(prefetch_dataset, nested_varinfo_variable)

            # Check that bounds variable exists in the nested group.
            self.assertTrue(nested_group.variables[nested_bounds_name])

            resulting_bounds_nested_data = nested_group.variables[
                nested_bounds_name][:]
            assert_array_equal(resulting_bounds_nested_data,
                               expected_bounds_data)
            # Check that varinfo variable has 'bounds' attribute.
            self.assertEqual(nested_varinfo_variable.attributes['bounds'],
                             nested_bounds_name)
            # Check that NetCDF4 dimension variable has 'bounds' attribute.
            self.assertEqual(nested_group.variables[
                nested_variable_name].__dict__.get('bounds'),
                nested_bounds_name)
            # Check that VariableFromDmr 'has bounds' reference in
            # the references dictionary.
            self.assertEqual(nested_varinfo_variable.references['bounds'],
                             {nested_bounds_name, })

    @patch('hoss.dimension_utilities.get_opendap_nc4')
    def test_prefetch_dimensions_with_bounds(self, mock_get_opendap_nc4):
        """ Ensure that a variable which has dimensions with `bounds` metadata
            retrieves both the dimension variables and the bounds variables to
            which their metadata refers.

        """
        prefetch_path = 'prefetch.nc4'
        mock_get_opendap_nc4.return_value = prefetch_path

        access_token = 'access'
        url = 'https://url_to_opendap_granule'
        required_variables = {'/Grid/precipitationCal', '/Grid/lat',
                              '/Grid/lon', '/Grid/time'}
        dimensions_and_bounds = {'/Grid/lat', '/Grid/lat_bnds', '/Grid/lon',
                                 '/Grid/lon_bnds', '/Grid/time',
                                 '/Grid/time_bnds'}

        self.assertEqual(prefetch_dimension_variables(url,
                                                      self.varinfo_with_bounds,
                                                      required_variables,
                                                      self.temp_dir,
                                                      self.logger,
                                                      access_token,
                                                      self.config),
                         prefetch_path)

        mock_get_opendap_nc4.assert_called_once_with(url,
                                                     dimensions_and_bounds,
                                                     self.temp_dir,
                                                     self.logger, access_token,
                                                     self.config)

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
        bounding_box = [10, 20, 30, 40]
        shape_file = {'href': 'path/to/shape.geo.json',
                      'type': 'application/geo+json'}
        temporal_range = {'start': '2021-01-01T01:30:00',
                          'end': '2021-01-01T02:00:00'}
        dimensions = [{'name': 'lev', 'min': 800, 'max': 900}]

        with self.subTest('Bounding box subset only'):
            self.assertTrue(is_index_subset(Message({
                'subset': {'bbox': bounding_box}
            })))

        with self.subTest('Named dimensions subset only'):
            self.assertTrue(is_index_subset(Message({
                'subset': {'dimensions': dimensions}
            })))

        with self.subTest('Shape file only'):
            self.assertTrue(is_index_subset(Message({
                'subset': {'shape': shape_file}
            })))

        with self.subTest('Temporal subset only'):
            self.assertTrue(is_index_subset(
                Message({'temporal': temporal_range}))
            )

        with self.subTest('Bounding box and temporal'):
            self.assertTrue(is_index_subset(Message({
                'subset': {'bbox': bounding_box},
                'temporal': temporal_range,
            })))

        with self.subTest('Shape file and temporal'):
            self.assertTrue(is_index_subset(Message({
                'subset': {'shape': shape_file},
                'temporal': temporal_range,
            })))

        with self.subTest('Bounding box and named dimension'):
            self.assertTrue(is_index_subset(Message({
                'subset': {'bbox': bounding_box, 'dimensions': dimensions}
            })))

        with self.subTest('Not an index range subset'):
            self.assertFalse(is_index_subset(Message({})))

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
            harmony_message = Message({
                'subset': {
                    'dimensions': [{'name': '/latitude', 'min': 20, 'max': 30}]
                }
            })
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           ascending_file, harmony_message),
                {'/latitude': (440, 479)}
            )

        with self.subTest('Multiple ascending dimensions'):
            # 20.0 ≤ latitude[440] ≤ 20.25, 29.75 ≤ latitude[479] ≤ 30.0
            # 140.0 ≤ longitude[560] ≤ 140.25, 149.75 ≤ longitude[599] ≤ 150.0
            harmony_message = Message({
                'subset': {
                    'dimensions': [{'name': '/latitude', 'min': 20, 'max': 30},
                                   {'name': '/longitude', 'min': 140, 'max': 150}]
                }
            })
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           ascending_file, harmony_message),
                {'/latitude': (440, 479), '/longitude': (560, 599)}
            )

        with self.subTest('Descending dimension'):
            # 30.0 ≥ latitude[240] ≥ 29.75, 20.25 ≥ latitude[279] ≥ 20.0
            harmony_message = Message({
                'subset': {
                    'dimensions': [{'name': '/latitude', 'min': 20, 'max': 30}]
                }
            })
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           descending_file, harmony_message),
                {'/latitude': (240, 279)}
            )

        with self.subTest('Dimension has no leading slash'):
            # 20.0 ≤ latitude[440] ≤ 20.25, 29.75 ≤ latitude[479] ≤ 30.0
            harmony_message = Message({
                'subset': {
                    'dimensions': [{'name': 'latitude', 'min': 20, 'max': 30}]
                }
            })
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           ascending_file, harmony_message),
                {'/latitude': (440, 479)}
            )

        with self.subTest('Unspecified minimum value'):
            # 29.75 ≤ latitude[479] ≤ 30.0
            harmony_message = Message({
                'subset': {'dimensions': [{'name': '/latitude', 'max': 30}]}
            })
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           ascending_file, harmony_message),
                {'/latitude': (0, 479)}
            )

        with self.subTest('Unspecified maximum value'):
            # 20.0 ≤ latitude[440] ≤ 20.25, 179.75 ≤ latitude[719] ≤ 180.0
            harmony_message = Message({
                'subset': {'dimensions': [{'name': '/latitude', 'min': 20}]}
            })
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           ascending_file, harmony_message),
                {'/latitude': (440, 719)}
            )

        with self.subTest('Descending, unspecified minimum value'):
            # 30.0 ≥ latitude[240] ≥ 29.75, 0.25 ≥ latitude[719] ≥ 0.0
            harmony_message = Message({
                'subset': {'dimensions': [{'name': '/latitude', 'max': 30}]}
            })
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           descending_file, harmony_message),
                {'/latitude': (240, 719)}
            )

        with self.subTest('Descending, unspecified maximum value'):
            # 20.25 ≥ latitude[279] ≥ 20.0
            harmony_message = Message({
                'subset': {'dimensions': [{'name': '/latitude', 'min': 20}]}
            })
            self.assertDictEqual(
                get_requested_index_ranges(required_variables, self.varinfo,
                                           descending_file, harmony_message),
                {'/latitude': (0, 279)}
            )

        with self.subTest('Unrecognised dimension'):
            # Check for a non-existent named dimension
            harmony_message = Message({
                'subset': {
                    'dimensions': [{'name': '/FooBar', 'min': None, 'max': 10}]
                }
            })
            with self.assertRaises(InvalidNamedDimension):
                get_requested_index_ranges(required_variables, self.varinfo,
                                           descending_file, harmony_message),

    @patch('hoss.dimension_utilities.get_dimension_index_range')
    def test_get_requested_index_ranges_bounds(self,
                                               mock_get_dimension_index_range):
        """ Ensure that if bounds are present for a dimension, they are used
            as an argument in the call to get_dimension_index_range.

        """
        mock_get_dimension_index_range.return_value = (2000, 2049)

        gpm_varinfo = VarInfoFromDmr('tests/data/GPM_3IMERGHH_example.dmr',
                                     short_name='GPM_3IMERGHH')
        gpm_prefetch_path = 'tests/data/GPM_3IMERGHH_prefetch.nc4'

        harmony_message = Message({'subset': {
            'dimensions': [{'name': '/Grid/lon', 'min': 20, 'max': 25}]
        }})

        self.assertDictEqual(
            get_requested_index_ranges({'/Grid/lon'}, gpm_varinfo,
                                       gpm_prefetch_path, harmony_message),
            {'/Grid/lon': (2000, 2049)}
        )
        mock_get_dimension_index_range.assert_called_once_with(ANY, 20, 25,
                                                               bounds_values=ANY)

        with Dataset(gpm_prefetch_path) as prefetch:
            assert_array_equal(
                mock_get_dimension_index_range.call_args_list[0][0][0],
                prefetch['/Grid/lon'][:]
            )
            assert_array_equal(
                mock_get_dimension_index_range.call_args_list[0][1]['bounds_values'],
                prefetch['/Grid/lon_bnds'][:]
            )

    @patch('hoss.dimension_utilities.get_dimension_indices_from_bounds')
    @patch('hoss.dimension_utilities.get_dimension_indices_from_values')
    def test_get_dimension_index_range_bounds(self,
                                              mock_get_indices_from_values,
                                              mock_get_indices_from_bounds):
        """ Ensure that the correct branch of the code is used depending on
            whether bounds are specified or not.

            Also ensure that the minimum and maximum requested extent are
            always in ascending order in calls to
            `get_dimension_indices_from_bounds`, regardless of if the
            dimension is ascending or descending.

        """
        dimension_values = np.ma.MaskedArray(np.linspace(0.5, 9.5, 10))

        lower_bounds = np.linspace(0, 9, 10)
        upper_bounds = np.linspace(1, 10, 10)
        dimension_bounds = np.ma.MaskedArray(np.array([lower_bounds,
                                                       upper_bounds]).T)

        with self.subTest('No bounds are specified'):
            get_dimension_index_range(dimension_values, 2.3, 4.6)
            mock_get_indices_from_values.assert_called_once_with(ANY, 2.3, 4.6)
            assert_array_equal(
                mock_get_indices_from_values.call_args_list[0][0][0],
                dimension_values
            )
            mock_get_indices_from_values.reset_mock()
            mock_get_indices_from_bounds.assert_not_called()

        with self.subTest('Bounds are specified'):
            get_dimension_index_range(dimension_values, 2.3, 4.6,
                                      dimension_bounds)
            mock_get_indices_from_values.assert_not_called()
            mock_get_indices_from_bounds.assert_called_once_with(ANY, 2.3, 4.6)
            assert_array_equal(
                mock_get_indices_from_bounds.call_args_list[0][0][0],
                dimension_bounds
            )
            mock_get_indices_from_bounds.reset_mock()

        with self.subTest('Bounds are specified, descending dimension'):
            get_dimension_index_range(np.flip(dimension_values), 2.3, 4.6,
                                      np.flip(dimension_bounds))
            mock_get_indices_from_values.assert_not_called()
            mock_get_indices_from_bounds.assert_called_once_with(ANY, 2.3, 4.6)
            assert_array_equal(
                mock_get_indices_from_bounds.call_args_list[0][0][0],
                np.flip(dimension_bounds)
            )
            mock_get_indices_from_bounds.reset_mock()

    def test_get_dimension_bounds(self):
        """ Ensure that if a dimension variable has a `bounds` metadata
            attribute, the values in the associated bounds variable are
            returned. Ensure graceful handling if the dimension variable lacks
            bounds metadata, or the referred to bounds variable is absent from
            the NetCDF-4 dataset.

        """
        with self.subTest('Bounds are retrieved'):
            with Dataset('tests/data/GPM_3IMERGHH_prefetch.nc4') as dataset:
                assert_array_equal(
                    get_dimension_bounds('/Grid/lat', self.varinfo_with_bounds,
                                         dataset),
                    dataset['/Grid/lat_bnds'][:]
                )

        with self.subTest('Variable has no bounds, None is returned'):
            with Dataset('tests/data/f16_ssmis_lat_lon.nc') as dataset:
                self.assertIsNone(get_dimension_bounds('/latitude',
                                                       self.varinfo, dataset))

        with self.subTest('Incorrect bounds metadata, None is returned'):
            prefetch_bad_bounds = f'{self.temp_dir}/f16_ssmis_lat_lon.nc'
            copy('tests/data/f16_ssmis_lat_lon.nc', prefetch_bad_bounds)

            with Dataset(prefetch_bad_bounds, 'r+') as dataset:
                dataset['/latitude'].setncattr('bounds', '/does_not_exist')
                self.assertIsNone(get_dimension_bounds('/latitude',
                                                       self.varinfo, dataset))

    def test_get_dimension_indices_from_bounds(self):
        """ Ensure that the correct index ranges are retrieved for a variety
            of requested dimension ranges, including values that lie within
            pixels and others on the boundary between two adjacent pixels.

        """
        ascending_bounds = np.array([[0, 10], [10, 20], [20, 30], [30, 40],
                                     [40, 50]])
        descending_bounds = np.array([[0, -10], [-10, -20], [-20, -30],
                                      [-30, -40], [-40, -50]])

        with self.subTest('Ascending dimension, values within pixels'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(ascending_bounds, 5, 15),
                (0, 1)
            )

        with self.subTest('Ascending dimension, min_value on pixel edge'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(ascending_bounds, 10, 15),
                (1, 1)
            )

        with self.subTest('Ascending dimension, max_value on pixel edge'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(ascending_bounds, 5, 20),
                (0, 1)
            )

        with self.subTest('Ascending dimension, min=max on pixel edge'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(ascending_bounds, 20, 20),
                (1, 2)
            )

        with self.subTest('Ascending dimension, min=max within a pixel'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(ascending_bounds, 15, 15),
                (1, 1)
            )

        with self.subTest('Ascending dimension, min_value < lowest bounds'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(ascending_bounds, -10, 15),
                (0, 1)
            )

        with self.subTest('Ascending dimension, max_value > highest bound'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(ascending_bounds, 45, 55),
                (4, 4)
            )

        with self.subTest('Ascending dimension, max_value < lowest bound'):
            with self.assertRaises(InvalidRequestedRange):
                get_dimension_indices_from_bounds(ascending_bounds, -15, -5)

        with self.subTest('Ascending dimension, min_value > highest bound'):
            with self.assertRaises(InvalidRequestedRange):
                get_dimension_indices_from_bounds(ascending_bounds, 55, 65)

        with self.subTest('Descending dimension, values within pixels'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(descending_bounds, -15, -5),
                (0, 1)
            )

        with self.subTest('Descending dimension, max_value on pixel edge'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(descending_bounds, -15, -10),
                (1, 1)
            )

        with self.subTest('Descending dimension, min_value on pixel edge'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(descending_bounds, -20, -5),
                (0, 1)
            )

        with self.subTest('Descending dimension, min=max on pixel edge'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(descending_bounds, -20, -20),
                (1, 2)
            )

        with self.subTest('Descending dimension, min=max within a pixel'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(descending_bounds, -15, -15),
                (1, 1)
            )

        with self.subTest('Descending dimension, max_value > highest bounds'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(descending_bounds, -15, 10),
                (0, 1)
            )

        with self.subTest('Descending dimension, min_value > lowest bound'):
            self.assertTupleEqual(
                get_dimension_indices_from_bounds(descending_bounds, -55, -45),
                (4, 4)
            )

        with self.subTest('Descending dimension, min_value > highest bound'):
            with self.assertRaises(InvalidRequestedRange):
                get_dimension_indices_from_bounds(descending_bounds, 5, 15)

        with self.subTest('Descending dimension, max_value > lowest bound'):
            with self.assertRaises(InvalidRequestedRange):
                get_dimension_indices_from_bounds(descending_bounds, -65, -55)

    def test_is_almost_in(self):
        """ Ensure that only values within an acceptable tolerance of data are
            determined to have nearby values within the input array.

        """
        test_array = np.linspace(0, 1, 1001)

        true_tests = [
            ['0.1, value in test_array', test_array, 0.1],
            ['0.01, value in test_array ', test_array, 0.01],
            ['0.001, value in test_array', test_array, 0.001],
            ['0.0000001, below tolerance rounds to zero', test_array, 0.0000001]
        ]
        false_tests = [
            ['0.0001 - not in array, above tolerance', test_array, 0.0001],
            ['0.00001 - not in array, above tolerance', test_array, 0.00001],
        ]

        for description, input_array, input_value in true_tests:
            with self.subTest(description):
                self.assertTrue(is_almost_in(input_value, input_array))

        for description, input_array, input_value in false_tests:
            with self.subTest(description):
                self.assertFalse(is_almost_in(input_value, input_array))
