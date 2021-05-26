from logging import getLogger
from os import makedirs
from os.path import exists
from shutil import copy, rmtree
from unittest import TestCase
from unittest.mock import patch

from netCDF4 import Dataset
import numpy as np

from harmony.util import config
from varinfo import VarInfoFromDmr

from pymods.geo_grid import (add_index_range, fill_variables,
                             get_bounding_box_longitudes,
                             get_dimension_index_ranges,
                             get_dimension_index_range, get_fill_slice,
                             get_geo_bounding_box_subset,
                             get_required_dimensions,
                             get_valid_longitude_range,
                             is_geographic_coordinate, is_latitude,
                             is_longitude, unwrap_longitude, wrap_longitude)


class TestGeoGrid(TestCase):
    """ A class for testing functions in the pymods.utilities module. """
    @classmethod
    def setUpClass(cls):
        cls.config = config(validate=False)
        cls.logger = getLogger('tests')
        cls.dataset = VarInfoFromDmr('tests/data/rssmif16d_example.dmr',
                                     cls.logger,
                                     'tests/data/test_subsetter_config.yml')
        cls.test_dir = 'tests/output'

    def tearDown(self):
        if exists(self.test_dir):
            rmtree(self.test_dir)

    @patch('pymods.geo_grid.fill_variables')
    @patch('pymods.geo_grid.get_dimension_index_ranges')
    @patch('pymods.geo_grid.get_opendap_nc4')
    def test_get_geo_bounding_box_subset(self, mock_get_opendap_nc4,
                                         mock_get_dimension_index_ranges,
                                         mock_fill_variables):
        """ Ensure a request to get a geographic bounding box will make the
            expected requests to retrieve OPeNDAP data. The filling should not
            be called as the bounding box does not cross the longitude
            discontinuity at the edge of the grid.

            This test is primarily a unit test to ensure the correct branches
            of code are exercised, calling other functions with the expected
            arguments. For an end-to-end test see: `tests/test_subsetter.py`.

        """
        access_token = '0p3n5354m3!'
        url = 'https://opendap.earthdata.nasa.gov/path/to/granule'
        test_file = 'tests/data/f16_ssmis_20200102v7.nc'
        makedirs(self.test_dir)
        copy(test_file, self.test_dir)

        bounding_box = [20, 20, 40, 40]
        required_variables = {'/rainfall_rate', '/latitude', '/longitude'}
        mock_get_opendap_nc4.side_effect = ['dimensions.nc', 'final.nc']
        mock_get_dimension_index_ranges.return_value = {'/longitude': [80, 160],
                                                        '/latitude': [440, 520]}

        self.assertEqual(
            get_geo_bounding_box_subset(required_variables, self.dataset,
                                        bounding_box, url, self.test_dir,
                                        self.logger, access_token,
                                        self.config),
            'final.nc'
        )

        self.assertEqual(mock_get_opendap_nc4.call_count, 2)
        mock_get_opendap_nc4.assert_any_call(
            url, {'/latitude', '/longitude'}, self.test_dir, self.logger,
            access_token, self.config
        )
        mock_get_opendap_nc4.assert_any_call(
            url,
            {'/rainfall_rate[][440:520][80:160]', '/longitude[80:160]',
             '/latitude[440:520]'},
            self.test_dir, self.logger, access_token, self.config
        )
        mock_get_dimension_index_ranges.assert_called_once_with(
            'dimensions.nc', self.dataset, {'/latitude', '/longitude'},
            bounding_box
        )
        mock_fill_variables.assert_not_called()

    @patch('pymods.geo_grid.fill_variables')
    @patch('pymods.geo_grid.get_dimension_index_ranges')
    @patch('pymods.geo_grid.get_opendap_nc4')
    def test_get_geo_bounding_box_grid_edge(self, mock_get_opendap_nc4,
                                            mock_get_dimension_index_ranges,
                                            mock_fill_variables):
        """ Ensure a request to get a geographic bounding box will make the
            expected requests to retrieve OPeNDAP data. Also ensure that
            because the bounding box crosses the edge of a grid, filling is
            attempted.

            This test is primarily a unit test to ensure the correct branches
            of code are exercised, calling other functions with the expected
            arguments. For an end-to-end test see: `tests/test_subsetter.py`.

        """
        access_token = '0p3n5354m3!'
        url = 'https://opendap.earthdata.nasa.gov/path/to/granule'
        test_file = 'tests/data/f16_ssmis_20200102v7.nc'
        makedirs(self.test_dir)
        copy(test_file, self.test_dir)

        bounding_box = [-20, 0, 10, 30]
        required_variables = {'/rainfall_rate', '/latitude', '/longitude'}
        mock_get_opendap_nc4.side_effect = ['dimensions.nc', 'final.nc']
        mock_get_dimension_index_ranges.return_value = {
            '/longitude': [1360, 40],
            '/latitude': [360, 480]
        }

        self.assertEqual(
            get_geo_bounding_box_subset(required_variables, self.dataset,
                                        bounding_box, url, self.test_dir,
                                        self.logger, access_token,
                                        self.config),
            'final.nc'
        )

        self.assertEqual(mock_get_opendap_nc4.call_count, 2)
        mock_get_opendap_nc4.assert_any_call(
            url, {'/latitude', '/longitude'}, self.test_dir, self.logger,
            access_token, self.config
        )
        mock_get_opendap_nc4.assert_any_call(
            url,
            {'/rainfall_rate[][360:480][]', '/longitude',
             '/latitude[360:480]'},
            self.test_dir, self.logger, access_token, self.config
        )
        mock_get_dimension_index_ranges.assert_called_once_with(
            'dimensions.nc', self.dataset, {'/latitude', '/longitude'},
            bounding_box
        )
        mock_fill_variables.assert_called_once_with('final.nc',
                                                    self.dataset,
                                                    required_variables,
                                                    {'/longitude': [1360, 40]})

    @patch('pymods.geo_grid.fill_variables')
    @patch('pymods.geo_grid.get_dimension_index_ranges')
    @patch('pymods.geo_grid.get_opendap_nc4')
    def test_get_geo_bounding_box_no_geo(self, mock_get_opendap_nc4,
                                         mock_get_dimension_index_ranges,
                                         mock_fill_variables):
        """ Ensure a request to get a geographic bounding box will make the
            expected requests to retrieve OPeNDAP data. In this request there
            are no requested geographic variables, so the function should
            request the full range for all requested variables.

            This test is primarily a unit test to ensure the correct branches
            of code are exercised, calling other functions with the expected
            arguments. For an end-to-end test see: `tests/test_subsetter.py`.

        """
        access_token = '0p3n5354m3!'
        url = 'https://opendap.earthdata.nasa.gov/path/to/granule'
        test_file = 'tests/data/f16_ssmis_20200102v7.nc'
        makedirs(self.test_dir)
        copy(test_file, self.test_dir)

        bounding_box = [-20, 0, 10, 30]
        required_variables = {'/time'}

        mock_get_opendap_nc4.return_value = 'final.nc'
        mock_get_dimension_index_ranges.return_value = {}

        self.assertEqual(
            get_geo_bounding_box_subset(required_variables, self.dataset,
                                        bounding_box, url, self.test_dir,
                                        self.logger, access_token,
                                        self.config),
            'final.nc'
        )

        mock_get_opendap_nc4.assert_called_once_with(
            url, {'/time'}, self.test_dir, self.logger, access_token,
            self.config
        )
        mock_get_dimension_index_ranges.assert_not_called()
        mock_fill_variables.assert_not_called()

    def test_get_required_dimensions(self):
        """ Ensure a set of required variables can be parsed and a set of
            dimensions used by all of those variables is returned.

        """
        with self.subTest('Combines different dimensions'):
            self.assertSetEqual(
                get_required_dimensions(self.dataset,
                                        {'/latitude', '/longitude'}),
                {'/latitude', '/longitude'}
            )

        with self.subTest('Has a single entry for repeated dimensions'):
            self.assertSetEqual(
                get_required_dimensions(self.dataset,
                                        {'/rainfall_rate', '/sst_dtime'}),
                {'/latitude', '/longitude', '/time'}
            )

    def test_get_dimension_index_ranges(self):
        """ Ensure that correct index ranges can be calculated for:

            - Latitude dimensions
            - Longitude dimensions (continuous ranges)
            - Longitude dimensions (bounding box crossing grid edge)

        """
        makedirs(self.test_dir)
        test_file_name = f'{self.test_dir}/test.nc'
        bounding_box = [-20, 45, 20, 85]

        with Dataset(test_file_name, 'w', format='NETCDF4') as test_file:
            test_file.createDimension('latitude', size=180)
            test_file.createDimension('longitude', size=360)

            test_file.createVariable('latitude', float,
                                     dimensions=('latitude', ))
            test_file['latitude'][:] = np.linspace(-89.5, 89.5, 180)
            test_file['latitude'].setncatts({'units': 'degrees_north'})

            test_file.createVariable('longitude', float,
                                     dimensions=('longitude', ))
            test_file['longitude'][:] = np.linspace(-179.5, 179.5, 360)
            test_file['longitude'].setncatts({'units': 'degrees_east'})

        with self.subTest('Latitude dimension'):
            # latitude[134] = 44.5, latitude[174] = 84.5
            self.assertDictEqual(
                get_dimension_index_ranges(test_file_name, self.dataset,
                                           {'/latitude'}, bounding_box),
                {'/latitude': [134, 174]}
            )

        with self.subTest('Longitude dimension, bounding box within grid'):
            self.assertDictEqual(
                get_dimension_index_ranges(test_file_name, self.dataset,
                                           {'/longitude'}, bounding_box),
                {'/longitude': [160, 200]}
            )

        with Dataset(test_file_name, 'a', format='NETCDF4') as test_file:
            test_file['longitude'][:] = np.linspace(0.5, 359.5, 360)

        with self.subTest('Longitude, bounding box crosses grid edge'):
            self.assertDictEqual(
                get_dimension_index_ranges(test_file_name, self.dataset,
                                           {'/longitude'}, bounding_box),
                {'/longitude': [340, 20]}
            )

    def test_get_dimension_index_range(self):
        """ Ensure the expected index values are retrieved for the minimum and
            maximum values of an expected range. This should correspond to the
            nearest integer, to ensure partial pixels are included in a
            bounding box spatial subset. List elements must be integers for
            later array slicing.

        """
        data = np.linspace(0, 200, 101)
        dimension_data = np.ma.masked_array(data=data, mask=False)
        minimum_extent = 39
        maximum_extent = 174.3

        # data[20] = 40.0, data[87] = 174.0
        expected_results = [20, 87]

        results = get_dimension_index_range(dimension_data, minimum_extent,
                                            maximum_extent)

        self.assertIsInstance(results[0], int)
        self.assertIsInstance(results[1], int)
        self.assertListEqual(results, expected_results)

    def test_get_bounding_box_longitudes(self):
        """ Ensure the western and eastern extents of a bounding box are
            converted to the correct range according to the range of the
            longitude variable.

            If the variable range is -180 ≤ longitude (degrees) < 180, then the
            bounding box values should remain un converted. If the variable
            range is 0 ≤ longitude (degrees) < 360, then the bounding box
            values should be converted to this range.

        """
        bounding_box = [-150, -15, -120, 15]

        dataset = Dataset('test.nc', 'w', diskless=True)
        dataset.createDimension('longitude', size=361)

        test_args = [
            ['-180 ≤ lon (deg) < 180', 'lon_wrap', -180, 180, [-150, -120]],
            ['0 ≤ lon (deg) < 360', 'lon_unwrap', 0, 360, [210, 240]],
        ]

        for description, var_name, valid_min, valid_max, results in test_args:
            with self.subTest(description):
                data = np.linspace(valid_min, valid_max, 361)
                longitude = dataset.createVariable(var_name, data.dtype,
                                                   dimensions=('longitude', ))
                longitude[:] = data
                longitude.setncatts({'valid_min': valid_min,
                                     'valid_max': valid_max})

                self.assertListEqual(
                    get_bounding_box_longitudes(bounding_box, dataset[var_name]),
                    results
                )

        dataset.close()

    def test_wrap_longitude(self):
        """ Ensure that longitudes are correctly mapped to the
            -180 ≤ longitude (degrees) < 180 range.

            `TestCase.assertAlmostEqual` rounds to 7 decimal places.

        """
        longitudes = [['Needs wrapping', 190.0, -170.0],
                      ['Already wrapped', 123.45, 123.45]]

        for description, longitude, expected_longitude in longitudes:
            with self.subTest(description):
                self.assertAlmostEqual(wrap_longitude(longitude),
                                       expected_longitude)

    def test_unwrap_longitudes(self):
        """ Ensure that longitudes are correctly mapped to the
            0 ≤ longitude (degrees) < 360 range.

        """
        longitudes = [['Needs unwrapping', -160.5, 199.5],
                      ['Already unwrapped', 12.34, 12.34]]

        for description, longitude, expected_longitude in longitudes:
            with self.subTest(description):
                self.assertAlmostEqual(unwrap_longitude(longitude),
                                       expected_longitude)

    def test_get_valid_longitude_range(self):
        """ Ensure the valid longitude can be extracted from either the
            valid_range or valid_min and valid_max metadata attributes. Ensure
            that, if these metadata attributes are absent, the longitude range
            can be identified from the data themselves.

        """
        unwrapped_data = np.linspace(0, 360, 361)
        wrapped_data = np.linspace(-180, 180, 361)

        with Dataset('tests.nc', 'w', diskless=True) as test_file:
            test_file.createDimension('longitude', size=361)

            with self.subTest('valid_range metadata attribute is  present'):
                variable = test_file.createVariable('valid_range',
                                                    unwrapped_data.dtype,
                                                    dimensions=('longitude', ))
                variable.setncatts({'valid_range': [0, 360]})
                self.assertListEqual(get_valid_longitude_range(variable),
                                     [0, 360])

            with self.subTest('valid_min and valid_max metadata attributes'):
                variable = test_file.createVariable('valid_min_max',
                                                    wrapped_data.dtype,
                                                    dimensions=('longitude', ))
                variable.setncatts({'valid_min': -180, 'valid_max': 180})
                self.assertListEqual(get_valid_longitude_range(variable),
                                     [-180, 180])

            with self.subTest('No metadata attributes, data > 180 degrees'):
                variable = test_file.createVariable('from_data_0_360',
                                                    unwrapped_data.dtype,
                                                    dimensions=('longitude', ))
                variable[:] = unwrapped_data
                self.assertListEqual(get_valid_longitude_range(variable),
                                     [0, 360])

            with self.subTest('No metadata attributes, data ≤ 180 degrees'):
                variable = test_file.createVariable('from_data_180_180',
                                                    wrapped_data.dtype,
                                                    dimensions=('longitude', ))
                variable[:] = wrapped_data
                self.assertListEqual(get_valid_longitude_range(variable),
                                     [-180, 180])

            with self.subTest('Only valid_min resorts to data range'):
                variable = test_file.createVariable('only_valid_min',
                                                    unwrapped_data.dtype,
                                                    dimensions=('longitude', ))
                variable.setncatts({'valid_min': 0})
                variable[:] = unwrapped_data
                self.assertListEqual(get_valid_longitude_range(variable),
                                     [0, 360])

            with self.subTest('Only valid_max resorts to data range'):
                variable = test_file.createVariable('only_valid_max',
                                                    unwrapped_data.dtype,
                                                    dimensions=('longitude', ))
                variable.setncatts({'valid_max': 360})
                variable[:] = unwrapped_data
                self.assertListEqual(get_valid_longitude_range(variable),
                                     [0, 360])

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
            self.assertEqual(add_index_range('/sst_dtime', self.dataset,
                                             index_ranges),
                             '/sst_dtime')

        with self.subTest('With index constraints'):
            index_ranges = {'/latitude': [12, 34], '/longitude': [45, 56]}
            self.assertEqual(add_index_range('/sst_dtime', self.dataset,
                                             index_ranges),
                             '/sst_dtime[][12:34][45:56]')

        with self.subTest('With a longitude crossing discontinuity'):
            index_ranges = {'/latitude': [12, 34], '/longitude': [56, 5]}
            self.assertEqual(add_index_range('/sst_dtime', self.dataset,
                                             index_ranges),
                             '/sst_dtime[][12:34][]')

    def test_fill_variables(self):
        """ Ensure only the expected variables are filled (e.g., those with
            a longitude crossing the grid edge). Longitude variables should not
            themselves be filled.

        """
        makedirs(self.test_dir)
        input_file = 'tests/data/f16_ssmis_20200102v7.nc'
        test_file = copy(input_file, self.test_dir)
        fill_ranges = {'/longitude': [1400, 10]}
        required_variables = {'/sst_dtime', '/wind_speed'}

        fill_variables(test_file, self.dataset, required_variables,
                       fill_ranges)

        with Dataset(test_file, 'r') as test_output, Dataset(input_file, 'r') as test_input:
            # Assert none of the dimension variables are filled at any pixel
            for variable_dimension in ['/time', '/latitude', '/longitude']:
                data = test_output[variable_dimension][:]
                self.assertFalse(np.any(data.mask))
                np.testing.assert_array_equal(test_input[variable_dimension],
                                              test_output[variable_dimension])

            # Assert the expected range of wind_speed and sst_dtime are filled
            # but that rest of the variable matches the input file.
            for variable in required_variables:
                input_data = test_input[variable][:]
                output_data = test_output[variable][:]
                self.assertTrue(np.all(output_data[:][:][11:1400].mask))
                np.testing.assert_array_equal(output_data[:][:][:11],
                                              input_data[:][:][:11])
                np.testing.assert_array_equal(output_data[:][:][1400:],
                                              input_data[:][:][1400:])

            # Assert a variable that wasn't to be filled isn't
            rainfall_rate_in = test_input['/rainfall_rate'][:]
            rainfall_rate_out = test_output['/rainfall_rate'][:]
            np.testing.assert_array_equal(rainfall_rate_in, rainfall_rate_out)

    def test_get_fill_slice(self):
        """ Ensure that a slice object is correctly formed for a requested
            dimension.

            Longitude variables should not be filled, to ensure that there are
            valid grid coordinates for science variables (sub-test 3).

        """
        fill_ranges = {'/longitude': [200, 15]}
        longitude = self.dataset.get_variable('/longitude')
        wind_speed = self.dataset.get_variable('/wind_speed')

        with self.subTest('An unfilled dimension returns slice(None).'):
            self.assertEqual(
                get_fill_slice(wind_speed, '/time', fill_ranges),
                slice(None)
            )

        with self.subTest('A filled dimension returns slice(start, stop).'):
            self.assertEqual(
                get_fill_slice(wind_speed, '/longitude', fill_ranges),
                slice(16, 200)
            )

        with self.subTest('A longitude variable returns slice(None).'):
            self.assertEqual(
                get_fill_slice(longitude, '/longitude', fill_ranges),
                slice(None)
            )

    def test_is_geographic_coordinate(self):
        """ Ensure a latitude and longitude can be correctly determined as
            geographic.

        """
        with self.subTest('A latitude variable returns True'):
            self.assertTrue(
                is_geographic_coordinate(self.dataset.get_variable('/latitude'))
            )

        with self.subTest('A longitude variable returns True'):
            self.assertTrue(
                is_geographic_coordinate(self.dataset.get_variable('/longitude'))
            )

        with self.subTest('A non geographic dimension returns False'):
            self.assertFalse(
                is_geographic_coordinate(self.dataset.get_variable('/sst_dtime'))
            )

    def test_is_latitude(self):
        """ Ensure a latitude variable is correctly identified, but that
            longitudes or other non-latitudes are not misattributed.

        """
        with self.subTest('A latitude variable returns True'):
            self.assertTrue(is_latitude(self.dataset.get_variable('/latitude')))

        with self.subTest('A longitude variable returns False'):
            self.assertFalse(is_latitude(self.dataset.get_variable('/longitude')))

        with self.subTest('A non geographic dimension returns False'):
            self.assertFalse(is_latitude(self.dataset.get_variable('/sst_dtime')))

    def test_is_longitude(self):
        """ Ensure a longitude variable is correctly identified, but that
            latitudes or other non-longitudes are not misattributed.

        """
        with self.subTest('A latitude variable returns False'):
            self.assertFalse(is_longitude(self.dataset.get_variable('/latitude')))

        with self.subTest('A longitude variable returns True'):
            self.assertTrue(is_longitude(self.dataset.get_variable('/longitude')))

        with self.subTest('A non geographic dimension returns False'):
            self.assertFalse(is_longitude(self.dataset.get_variable('/sst_dtime')))
