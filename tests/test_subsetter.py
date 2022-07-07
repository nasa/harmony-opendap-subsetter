from shutil import copy, rmtree
from tempfile import mkdtemp
from typing import Dict, Set
from unittest import TestCase
from unittest.mock import ANY, Mock, patch

from harmony.message import Message
from harmony.util import config, HarmonyException
from netCDF4 import Dataset
from numpy.testing import assert_array_equal

from subsetter import HarmonyAdapter
from tests.utilities import write_dmr


class TestSubsetterEndToEnd(TestCase):

    @classmethod
    def setUpClass(cls):
        """ Test fixture that can be set once for all tests in the class. """
        cls.granule_url = 'https://harmony.uat.earthdata.nasa.gov/opendap_url'
        cls.atl03_variable = '/gt1r/geophys_corr/geoid'
        cls.rssmif16d_variable = '/wind_speed'

        with open('tests/data/ATL03_example.dmr', 'r') as file_handler:
            cls.atl03_dmr = file_handler.read()

        with open('tests/data/rssmif16d_example.dmr', 'r') as file_handler:
            cls.rssmif16d_dmr = file_handler.read()

        with open('tests/data/M2T1NXSLV_example.dmr', 'r') as file_handler:
            cls.m2t1nxslv_dmr = file_handler.read()

    def setUp(self):
        """ Have to mock mkdtemp, to know where to put mock .dmr content. """
        self.tmp_dir = mkdtemp()
        self.config = config(validate=False)

    def tearDown(self):
        rmtree(self.tmp_dir)

    def assert_valid_request_data(self, request_data: Dict,
                                  expected_variables: Set[str]):
        """ Check the contents of the request data sent to the OPeNDAP server
            when retrieving a NetCDF-4 file. This should ensure that a URL
            encoded constraint expression was sent, and that all the expected
            variables (potentially with index ranges) were included.

            This custom class method is used because the contraint expressions
            are constructed from sets. The order of variables in the set, and
            therefore the constraint expression string, cannot be guaranteed.

        """
        opendap_separator = '%3B'
        self.assertIn('dap4.ce', request_data)
        requested_variables = set(request_data['dap4.ce'].split(opendap_separator))
        self.assertSetEqual(requested_variables, expected_variables)

    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_non_geo_end_to_end(self, mock_stage, mock_util_download,
                                mock_rmtree, mock_mkdtemp, mock_uuid):
        """ Ensure the subsetter will run end-to-end, only mocking the
            HTTP responses, and the output interactions with Harmony.

        """
        mock_uuid.return_value = Mock(hex='uuid')
        mock_mkdtemp.return_value = self.tmp_dir
        dmr_path = write_dmr(self.tmp_dir, self.atl03_dmr)

        downloaded_nc4_path = f'{self.tmp_dir}/opendap_url_subset.nc4'
        # There needs to be a physical file present to be renamed by Harmony.
        # The contents are not accessed.
        copy('tests/data/ATL03_example.dmr', downloaded_nc4_path)

        mock_util_download.side_effect = [dmr_path, downloaded_nc4_path]

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2020-01-01T00:00:00.000Z',
                        'end': '2020-01-02T00:00:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
                'variables': [{'id': '',
                               'name': self.atl03_variable,
                               'fullPath': self.atl03_variable}]}],
            'stagingLocation': 's3://example-bucket/',
            'user': 'fhaise',
        }
        message = Message(message_data)

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should be the required
        # variables. (There is no dimension prefetch in this type of request)
        self.assertEqual(mock_util_download.call_count, 2)
        mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=None,
                                           cfg=subsetter.config)
        mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=ANY,
                                           cfg=subsetter.config)

        # Ensure the constraint expression contains all the required variables.
        post_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            post_data,
            {'%2Fgt1r%2Fgeolocation%2Fdelta_time',
             '%2Fgt1r%2Fgeolocation%2Freference_photon_lon',
             '%2Fgt1r%2Fgeolocation%2Fpodppd_flag',
             '%2Fgt1r%2Fgeophys_corr%2Fdelta_time',
             '%2Fgt1r%2Fgeolocation%2Freference_photon_lat',
             '%2Fgt1r%2Fgeophys_corr%2Fgeoid'}
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid.nc4',
            'opendap_url_gt1r_geophys_corr_geoid_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

    @patch('pymods.dimension_utilities.get_fill_slice')
    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_geo_end_to_end(self, mock_stage, mock_util_download, mock_rmtree,
                            mock_mkdtemp, mock_uuid, mock_get_fill_slice):
        """ Ensure a request with a bounding box will be correctly processed,
            requesting only the expected variables, with index ranges
            corresponding to the bounding box specified.

        """
        bounding_box = [-30, 45, -15, 60]

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path,
                                          all_variables_path]

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2020-01-01T00:00:00.000Z',
                        'end': '2020-01-02T00:00:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
                'variables': [{'id': '',
                               'name': self.rssmif16d_variable,
                               'fullPath': self.rssmif16d_variable}]}],
            'stagingLocation': 's3://example-bucket/',
            'subset': {'bbox': bounding_box},
            'user': 'jlovell',
        }
        message = Message(message_data)

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should fetch a NetCDF-4
        # file containing the full 1-D dimension variables only, and the third
        # should retrieve the final NetCDF-4 with all required variables.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=None,
                                           cfg=subsetter.config)

        # Because of the `ANY` match for the request data, the requests for
        # dimensions and all variables will look the same.
        mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=ANY,
                                           cfg=subsetter.config)

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )
        # Ensure the constraint expression contains all the required variables.
        # /wind_speed[][540:599][1320:1379], /time, /longitude[1320:1379]
        # /latitude[540:599]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {'%2Ftime',
             '%2Flatitude%5B540%3A599%5D',
             '%2Flongitude%5B1320%3A1379%5D',
             '%2Fwind_speed%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D'}
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            'opendap_url_wind_speed_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('pymods.dimension_utilities.get_fill_slice')
    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_geo_descending_latitude(self, mock_stage, mock_util_download,
                                     mock_rmtree, mock_mkdtemp, mock_uuid,
                                     mock_get_fill_slice):
        """ Ensure a request with a bounding box will be correctly processed,
            requesting only the expected variables, with index ranges
            corresponding to the bounding box specified. The latitude dimension
            returned from the geographic dimensions request to OPeNDAP will be
            descending. This test is to ensure the correct dimension indices
            are identified and the correct DAP4 constraint expression is built.

        """
        bounding_box = [-30, 45, -15, 60]

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon_desc.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo_desc.nc', all_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path,
                                          all_variables_path]

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2020-01-01T00:00:00.000Z',
                        'end': '2020-01-02T00:00:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
                'variables': [{'id': '',
                               'name': self.rssmif16d_variable,
                               'fullPath': self.rssmif16d_variable}]}],
            'stagingLocation': 's3://example-bucket/',
            'subset': {'bbox': bounding_box},
            'user': 'cduke',
        }
        message = Message(message_data)

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should fetch a NetCDF-4
        # file containing the full 1-D dimension variables only, and the third
        # should retrieve the final NetCDF-4 with all required variables.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=None,
                                           cfg=subsetter.config)

        # Because of the `ANY` match for the request data, the requests for
        # dimensions and all variables will look the same.
        mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=ANY,
                                           cfg=subsetter.config)

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'})
        # Ensure the constraint expression contains all the required variables.
        # /wind_speed[][120:179][1320:1379], /time, /longitude[1320:1379]
        # /latitude[120:179]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {'%2Ftime',
             '%2Flatitude%5B120%3A179%5D',
             '%2Flongitude%5B1320%3A1379%5D',
             '%2Fwind_speed%5B%5D%5B120%3A179%5D%5B1320%3A1379%5D'}
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            'opendap_url_wind_speed_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled:
        mock_get_fill_slice.assert_not_called()

    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_geo_crossing_grid_edge(self, mock_stage, mock_util_download,
                                    mock_rmtree, mock_mkdtemp, mock_uuid):
        """ Ensure a request with a bounding box that crosses a longitude edge
            (360 degrees east) requests the expected variables from OPeNDAP and
            does so only in the expected latitude range. The full longitude
            range should be requested for all variables, with filling applied
            outside of the bounding box region.

        """
        bounding_box = [-7.5, -60, 7.5, -45]

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        unfilled_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_unfilled.nc', unfilled_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path,
                                          unfilled_path]

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2020-01-01T00:00:00.000Z',
                        'end': '2020-01-02T00:00:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
                'variables': [{'id': '',
                               'name': self.rssmif16d_variable,
                               'fullPath': self.rssmif16d_variable}]}],
            'stagingLocation': 's3://example-bucket/',
            'subset': {'bbox': bounding_box},
            'user': 'jswiggert',
        }
        message = Message(message_data)

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should fetch a NetCDF-4
        # file containing the full 1-D dimension variables only, and the third
        # should retrieve the final NetCDF-4 with all required variables.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=None,
                                           cfg=subsetter.config)

        # Because of the `ANY` match for the request data, the requests for
        # dimensions and all variables will look the same.
        mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=ANY,
                                           cfg=subsetter.config)

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )
        # Ensure the constraint expression contains all the required variables.
        # /wind_speed[][120:179][], /time, /longitude (full range),
        # /latitude[120:179]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {'%2Ftime',
             '%2Flatitude%5B120%3A179%5D',
             '%2Flongitude',
             '%2Fwind_speed%5B%5D%5B120%3A179%5D%5B%5D'}
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            'opendap_url_wind_speed_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure the final output was correctly filled (the unfilled file is
        # filled in place):
        expected_output = Dataset('tests/data/f16_ssmis_filled.nc', 'r')
        actual_output = Dataset(f'{self.tmp_dir}/uuid2.nc4', 'r')

        for variable_name, expected_variable in expected_output.variables.items():
            self.assertIn(variable_name, actual_output.variables)
            assert_array_equal(actual_output[variable_name][:],
                               expected_variable[:])

        expected_output.close()
        actual_output.close()

    @patch('pymods.dimension_utilities.get_fill_slice')
    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_geo_bbox(self, mock_stage, mock_util_download, mock_rmtree,
                      mock_mkdtemp, mock_uuid, mock_get_fill_slice):
        """ Ensure requests with particular bounding box edge-cases return the
            correct pixel ranges:

            * Single point, N=S, W=E, inside a pixel, retrieves that single
              pixel.
            * Single point, N=S, W=E, in corner of 4 pixels retrieves all 4
              surrounding pixels.
            * Line, N=S, W < E, where the latitude is inside a pixel, retrieves
              a single row of pixels.
            * Line, N > S, W=E, where longitude is between pixels, retrieves
              two columns of pixels, corresponding to those which touch the
              line.

        """
        point_in_pixel = [-29.99, 45.01, -29.99, 45.01]
        point_between_pixels = [-30, 45, -30, 45]
        line_in_pixels = [-30, -14.99, -15, -14.99]
        line_between_pixels = [-30, 45, -30, 60]

        range_point_in_pixel = {
            '%2Ftime', '%2Flatitude%5B540%3A540%5D',
            '%2Flongitude%5B1320%3A1320%5D',
            '%2Fwind_speed%5B%5D%5B540%3A540%5D%5B1320%3A1320%5D'
        }

        range_point_between_pixels = {
            '%2Ftime', '%2Flatitude%5B539%3A540%5D',
            '%2Flongitude%5B1319%3A1320%5D',
            '%2Fwind_speed%5B%5D%5B539%3A540%5D%5B1319%3A1320%5D'
        }

        range_line_in_pixels = {
            '%2Ftime', '%2Flatitude%5B300%3A300%5D',
            '%2Flongitude%5B1320%3A1379%5D',
            '%2Fwind_speed%5B%5D%5B300%3A300%5D%5B1320%3A1379%5D'
        }

        range_line_between_pixels = {
            '%2Ftime', '%2Flatitude%5B540%3A599%5D',
            '%2Flongitude%5B1319%3A1320%5D',
            '%2Fwind_speed%5B%5D%5B540%3A599%5D%5B1319%3A1320%5D'
        }

        test_args = [['Point is inside single pixel', point_in_pixel,
                      range_point_in_pixel],
                     ['Point in corner of 4 pixels', point_between_pixels,
                      range_point_between_pixels],
                     ['Line through single row', line_in_pixels,
                      range_line_in_pixels],
                     ['Line between two columns', line_between_pixels,
                      range_line_between_pixels]]

        for description, bounding_box, expected_index_ranges in test_args:
            with self.subTest(description):
                mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
                mock_mkdtemp.return_value = self.tmp_dir

                dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

                dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
                copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

                all_variables_path = f'{self.tmp_dir}/variables.nc4'
                copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

                mock_util_download.side_effect = [dmr_path, dimensions_path,
                                                  all_variables_path]

                message_data = {
                    'accessToken': 'fake-token',
                    'callback': 'https://example.com/',
                    'sources': [{
                        'granules': [{
                            'id': 'G000-TEST',
                            'url': self.granule_url,
                            'temporal': {
                                'start': '2020-01-01T00:00:00.000Z',
                                'end': '2020-01-02T00:00:00.000Z'
                            },
                            'bbox': [-180, -90, 180, 90]
                        }],
                        'variables': [{'id': '',
                                       'name': self.rssmif16d_variable,
                                       'fullPath': self.rssmif16d_variable}]}],
                    'stagingLocation': 's3://example-bucket/',
                    'subset': {'bbox': bounding_box},
                    'user': 'jaaron',
                }
                message = Message(message_data)

                subsetter = HarmonyAdapter(message, config=config(False))
                subsetter.invoke()

                # Ensure the correct number of downloads were requested from
                # OPeNDAP: the first should be the `.dmr`. The second should
                # fetch a NetCDF-4 file containing the full 1-D dimension
                # variables only, and the third should retrieve the final
                # NetCDF-4 with all required variables.
                self.assertEqual(mock_util_download.call_count, 3)
                mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                                   self.tmp_dir,
                                                   subsetter.logger,
                                                   access_token=message_data['accessToken'],
                                                   data=None,
                                                   cfg=subsetter.config)

                # Because of the `ANY` match for the request data, the requests
                # for dimensions and all variables will look the same.
                mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                                   self.tmp_dir,
                                                   subsetter.logger,
                                                   access_token=message_data['accessToken'],
                                                   data=ANY,
                                                   cfg=subsetter.config)

                # Ensure the constraint expression for dimensions data included
                # only geographic or temporal variables with no index ranges
                dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
                self.assert_valid_request_data(
                    dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
                )
                # Ensure the constraint expression contains all the required variables.
                index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
                self.assert_valid_request_data(index_range_data,
                                               expected_index_ranges)

                # Ensure the output was staged with the expected file name
                mock_stage.assert_called_once_with(
                    f'{self.tmp_dir}/uuid2.nc4',
                    'opendap_url_wind_speed_subsetted.nc4',
                    'application/x-netcdf4',
                    location='s3://example-bucket/',
                    logger=subsetter.logger
                )
                mock_rmtree.assert_called_once_with(self.tmp_dir)

                # Ensure no variables were filled
                mock_get_fill_slice.assert_not_called()

            mock_mkdtemp.reset_mock()
            mock_uuid.reset_mock()
            mock_util_download.reset_mock()
            mock_stage.reset_mock()
            mock_get_fill_slice.reset_mock()
            mock_rmtree.reset_mock()

    @patch('pymods.dimension_utilities.get_fill_slice')
    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_geo_no_variables(self, mock_stage, mock_util_download,
                              mock_rmtree, mock_mkdtemp, mock_uuid,
                              mock_get_fill_slice):
        """ Ensure a request with a bounding box that does not specify any
            variables will retrieve all variables, but limited to the range
            specified by the bounding box.

        """
        bounding_box = [-30, 45, -15, 60]

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo_no_vars.nc', all_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path,
                                          all_variables_path]

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2020-01-01T00:00:00.000Z',
                        'end': '2020-01-02T00:00:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
                'variables': []
            }],
            'stagingLocation': 's3://example-bucket/',
            'subset': {'bbox': bounding_box},
            'user': 'kerwinj',
        }
        message = Message(message_data)

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should fetch a NetCDF-4
        # file containing the full 1-D dimension variables only, and the third
        # should retrieve the final NetCDF-4 with all required variables.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=None,
                                           cfg=subsetter.config)

        # Because of the `ANY` match for the request data, the requests for
        # dimensions and all variables will look the same.
        mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=ANY,
                                           cfg=subsetter.config)

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )
        # Ensure the constraint expression contains all variables.
        # /atmosphere_cloud_liquid_water_content[][540:599][1320:1379],
        # /atmosphere_water_vapor_content[][540:599][1320:1379],
        # /rainfall_rate[][540:599][1320:1379],
        # /sst_dtime[][540:599][1320:1379], /wind_speed[][540:599][1320:1379],
        # /time, /longitude[1320:1379], /latitude[540:599]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {'%2Ftime',
             '%2Flatitude%5B540%3A599%5D',
             '%2Flongitude%5B1320%3A1379%5D',
             '%2Fatmosphere_cloud_liquid_water_content%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D',
             '%2Fatmosphere_water_vapor_content%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D',
             '%2Frainfall_rate%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D',
             '%2Fsst_dtime%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D',
             '%2Fwind_speed%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D'}
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(f'{self.tmp_dir}/uuid2.nc4',
                                           'opendap_url_subsetted.nc4',
                                           'application/x-netcdf4',
                                           location='s3://example-bucket/',
                                           logger=subsetter.logger)
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled:
        mock_get_fill_slice.assert_not_called()

    @patch('pymods.dimension_utilities.get_fill_slice')
    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_temporal_end_to_end(self, mock_stage, mock_util_download,
                                 mock_rmtree, mock_mkdtemp, mock_uuid,
                                 mock_get_fill_slice):
        """ Ensure a request with a temporal range will retrieve variables,
            but limited to the range specified by the temporal range.

            The example granule has 24 hourly time slices, starting with
            2021-01-10T00:30:00.

        """
        temporal_range = {'start': '2021-01-10T01:00:00',
                          'end': '2021-01-10T03:00:00'}

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir

        dmr_path = write_dmr(self.tmp_dir, self.m2t1nxslv_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/M2T1NXSLV_prefetch.nc4', dimensions_path)

        temporal_variables_path = f'{self.tmp_dir}/temporal_variables.nc4'
        copy('tests/data/M2T1NXSLV_temporal.nc4', temporal_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path,
                                          temporal_variables_path]

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2021-01-10T00:30:00.000Z',
                        'end': '2021-01-11T00:30:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
                'variables': [{'id': '',
                               'name': '/PS',
                               'fullPath': '/PS'}]}],
            'stagingLocation': 's3://example-bucket/',
            'subset': None,
            'temporal': temporal_range,
            'user': 'jyoung',
        }
        message = Message(message_data)

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should fetch a NetCDF-4
        # file containing the full 1-D dimension variables only, and the third
        # should retrieve the final NetCDF-4 with all required variables.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=None,
                                           cfg=subsetter.config)

        # Because of the `ANY` match for the request data, the requests for
        # dimensions and all variables will look the same.
        mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=ANY,
                                           cfg=subsetter.config)

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(dimensions_data,
                                       {'%2Flat', '%2Flon', '%2Ftime'})
        # Ensure the constraint expression contains all the required variables.
        # /PS[1:2][][], /time[1:2], /lon, /lat
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {'%2Ftime%5B1%3A2%5D',
             '%2Flat',
             '%2Flon',
             '%2FPS%5B1%3A2%5D%5B%5D%5B%5D'}
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            'opendap_url_PS_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('pymods.dimension_utilities.get_fill_slice')
    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_temporal_all_variables(self, mock_stage, mock_util_download,
                                    mock_rmtree, mock_mkdtemp, mock_uuid,
                                    mock_get_fill_slice):
        """ Ensure a request with a temporal range and no specified variables
            will retrieve the expected output. Note - because a temporal range
            is specified, HOSS will need to perform an index range subset. This
            means that the prefetch will still have to occur, and all variables
            with the temporal grid dimension will need to include their index
            ranges in the final DAP4 constraint expression.

            The example granule has 24 hourly time slices, starting with
            2021-01-10T00:30:00.

        """
        temporal_range = {'start': '2021-01-10T01:00:00',
                          'end': '2021-01-10T03:00:00'}

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir

        dmr_path = write_dmr(self.tmp_dir, self.m2t1nxslv_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/M2T1NXSLV_prefetch.nc4', dimensions_path)

        temporal_variables_path = f'{self.tmp_dir}/temporal_variables.nc4'
        copy('tests/data/M2T1NXSLV_temporal.nc4', temporal_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path,
                                          temporal_variables_path]

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2021-01-10T00:30:00.000Z',
                        'end': '2021-01-11T00:30:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
            }],
            'stagingLocation': 's3://example-bucket/',
            'subset': None,
            'temporal': temporal_range,
            'user': 'jyoung',
        }
        message = Message(message_data)

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should fetch a NetCDF-4
        # file containing the full 1-D dimension variables only, and the third
        # should retrieve the final NetCDF-4 with all required variables.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=None,
                                           cfg=subsetter.config)

        # Because of the `ANY` match for the request data, the requests for
        # dimensions and all variables will look the same.
        mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=ANY,
                                           cfg=subsetter.config)

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(dimensions_data,
                                       {'%2Flat', '%2Flon', '%2Ftime'})
        # Ensure the constraint expression contains all the required variables.
        # /<science_variable>[1:2][][], /time[1:2], /lon, /lat
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {'%2Ftime%5B1%3A2%5D',
             '%2Flat',
             '%2Flon',
             '%2FCLDPRS%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FCLDTMP%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FDISPH%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FH1000%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FH250%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FH500%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FH850%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FPBLTOP%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FPS%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FOMEGA500%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FQ250%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FQ500%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FQ850%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FQV10M%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FQV2M%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FSLP%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FT10M%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FT250%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FT2M%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FT2MDEW%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FT2MWET%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FT500%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FT850%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FTO3%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FTOX%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FTQL%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FTQI%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FTQV%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FTROPPB%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FTROPPV%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FTROPQ%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FTROPT%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FTROPPT%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FTS%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FU10M%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FU250%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FU2M%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FU500%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FU50M%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FU850%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FV10M%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FV250%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FV2M%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FV500%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FV50M%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FV850%5B1%3A2%5D%5B%5D%5B%5D',
             '%2FZLCL%5B1%3A2%5D%5B%5D%5B%5D'}
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            'opendap_url_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('pymods.dimension_utilities.get_fill_slice')
    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_geo_temporal_end_to_end(self, mock_stage, mock_util_download,
                                     mock_rmtree, mock_mkdtemp, mock_uuid,
                                     mock_get_fill_slice):
        """ Ensure a request with both a bounding box and a temporal range will
            retrieve variables, but limited to the ranges specified by the
            bounding box and the temporal range.

        """
        temporal_range = {'start': '2021-01-10T01:00:00',
                          'end': '2021-01-10T03:00:00'}
        bounding_box = [40, -30, 50, -20]

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir

        dmr_path = write_dmr(self.tmp_dir, self.m2t1nxslv_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/M2T1NXSLV_prefetch.nc4', dimensions_path)

        geo_temporal_path = f'{self.tmp_dir}/geo_temporal.nc4'
        copy('tests/data/M2T1NXSLV_temporal.nc4', geo_temporal_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path,
                                          geo_temporal_path]

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2021-01-10T00:30:00.000Z',
                        'end': '2021-01-11T00:30:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
                'variables': [{'id': '',
                               'name': '/PS',
                               'fullPath': '/PS'}]}],
            'stagingLocation': 's3://example-bucket/',
            'subset': {'bbox': bounding_box},
            'temporal': temporal_range,
            'user': 'jyoung',
        }
        message = Message(message_data)

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should fetch a NetCDF-4
        # file containing the full 1-D dimension variables only, and the third
        # should retrieve the final NetCDF-4 with all required variables.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=None,
                                           cfg=subsetter.config)

        # Because of the `ANY` match for the request data, the requests for
        # dimensions and all variables will look the same.
        mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=ANY,
                                           cfg=subsetter.config)

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(dimensions_data,
                                       {'%2Flat', '%2Flon', '%2Ftime'})
        # Ensure the constraint expression contains all the required variables.
        # /PS[1:2][120:140][352:368], /time[1:2], /lon[352:368], /lat[120:140]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {'%2Ftime%5B1%3A2%5D',
             '%2Flat%5B120%3A140%5D',
             '%2Flon%5B352%3A368%5D',
             '%2FPS%5B1%3A2%5D%5B120%3A140%5D%5B352%3A368%5D'}
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            'opendap_url_PS_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('pymods.dimension_utilities.get_fill_slice')
    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.bbox_utilities.download')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_geo_shapefile_end_to_end(self, mock_stage, mock_util_download,
                                      mock_geojson_download, mock_rmtree,
                                      mock_mkdtemp, mock_uuid,
                                      mock_get_fill_slice):
        """ Ensure a request with a shape file specified will retrieve
            variables, but limited to the ranges of a bounding box that
            encloses the specified GeoJSON shape.

        """
        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

        geojson_path = f'{self.tmp_dir}/polygon.geo.json'
        copy('tests/geojson_examples/polygon.geo.json', geojson_path)

        shape_file_url = 'www.example.com/polygon.geo.json'
        mock_geojson_download.return_value = geojson_path
        mock_util_download.side_effect = [dmr_path, dimensions_path,
                                          all_variables_path]

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2020-01-01T00:00:00.000Z',
                        'end': '2020-01-02T00:00:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
                'variables': [{'id': '',
                               'name': self.rssmif16d_variable,
                               'fullPath': self.rssmif16d_variable}]}],
            'stagingLocation': 's3://example-bucket/',
            'subset': {'shape': {'href': shape_file_url,
                                 'type': 'application/geo+json'}},
            'user': 'dscott',
        }
        message = Message(message_data)

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        # Ensure the shape file in the Harmony message was downloaded:
        mock_geojson_download.assert_called_once_with(shape_file_url,
                                                      self.tmp_dir,
                                                      logger=subsetter.logger,
                                                      access_token=message_data['accessToken'],
                                                      cfg=subsetter.config)

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should fetch a NetCDF-4
        # file containing the full 1-D dimension variables only, and the third
        # should retrieve the final NetCDF-4 with all required variables.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=None,
                                           cfg=subsetter.config)

        # Because of the `ANY` match for the request data, the requests for
        # dimensions and all variables will look the same.
        mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=ANY,
                                           cfg=subsetter.config)

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )

        # Ensure the constraint expression contains all the required variables;
        # the polygon in the GeoJSON is the state of Utah:
        # /wind_speed[][508:527][983:1003], /time, /longitude[983:1003]
        # /latitude[508:527]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {'%2Ftime',
             '%2Flatitude%5B508%3A527%5D',
             '%2Flongitude%5B983%3A1003%5D',
             '%2Fwind_speed%5B%5D%5B508%3A527%5D%5B983%3A1003%5D'}
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            'opendap_url_wind_speed_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('pymods.dimension_utilities.get_fill_slice')
    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.bbox_utilities.download')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_geo_shapefile_all_variables(self, mock_stage, mock_util_download,
                                         mock_geojson_download, mock_rmtree,
                                         mock_mkdtemp, mock_uuid,
                                         mock_get_fill_slice):
        """ Ensure an all variable request with a shape file specified will
            retrieve all variables, but limited to the ranges of a bounding box
            that encloses the specified GeoJSON shape.

            Because a shape file is specified, index range subsetting will be
            performed, so a prefetch request will be performed, and the final
            DAP4 constraint expression will include all variables with index
            ranges.

        """
        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

        geojson_path = f'{self.tmp_dir}/polygon.geo.json'
        copy('tests/geojson_examples/polygon.geo.json', geojson_path)

        shape_file_url = 'www.example.com/polygon.geo.json'
        mock_geojson_download.return_value = geojson_path
        mock_util_download.side_effect = [dmr_path, dimensions_path,
                                          all_variables_path]

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2020-01-01T00:00:00.000Z',
                        'end': '2020-01-02T00:00:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
            }],
            'stagingLocation': 's3://example-bucket/',
            'subset': {'shape': {'href': shape_file_url,
                                 'type': 'application/geo+json'}},
            'user': 'dscott',
        }
        message = Message(message_data)

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        # Ensure the shape file in the Harmony message was downloaded:
        mock_geojson_download.assert_called_once_with(shape_file_url,
                                                      self.tmp_dir,
                                                      logger=subsetter.logger,
                                                      access_token=message_data['accessToken'],
                                                      cfg=subsetter.config)

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should fetch a NetCDF-4
        # file containing the full 1-D dimension variables only, and the third
        # should retrieve the final NetCDF-4 with all required variables.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=None,
                                           cfg=subsetter.config)

        # Because of the `ANY` match for the request data, the requests for
        # dimensions and all variables will look the same.
        mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=ANY,
                                           cfg=subsetter.config)

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )

        # Ensure the constraint expression contains all the required variables;
        # the polygon in the GeoJSON is the state of Utah:
        # /<science_variable>[][508:527][983:1003], /time, /longitude[983:1003]
        # /latitude[508:527]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {'%2Ftime',
             '%2Flatitude%5B508%3A527%5D',
             '%2Flongitude%5B983%3A1003%5D',
             '%2Fatmosphere_cloud_liquid_water_content%5B%5D%5B508%3A527%5D%5B983%3A1003%5D',
             '%2Fatmosphere_water_vapor_content%5B%5D%5B508%3A527%5D%5B983%3A1003%5D',
             '%2Frainfall_rate%5B%5D%5B508%3A527%5D%5B983%3A1003%5D',
             '%2Fsst_dtime%5B%5D%5B508%3A527%5D%5B983%3A1003%5D',
             '%2Fwind_speed%5B%5D%5B508%3A527%5D%5B983%3A1003%5D'}
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            'opendap_url_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('pymods.dimension_utilities.get_fill_slice')
    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.bbox_utilities.download')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_bbox_precedence_end_to_end(self, mock_stage, mock_util_download,
                                        mock_geojson_download, mock_rmtree,
                                        mock_mkdtemp, mock_uuid,
                                        mock_get_fill_slice):
        """ Ensure a request with a bounding box will be correctly processed,
            requesting only the expected variables, with index ranges
            corresponding to the bounding box specified.

        """
        bounding_box = [-30, 45, -15, 60]

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

        geojson_path = f'{self.tmp_dir}/polygon.geo.json'
        copy('tests/geojson_examples/polygon.geo.json', geojson_path)

        shape_file_url = 'www.example.com/polygon.geo.json'
        mock_geojson_download.return_value = geojson_path
        mock_util_download.side_effect = [dmr_path, dimensions_path,
                                          all_variables_path]

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2020-01-01T00:00:00.000Z',
                        'end': '2020-01-02T00:00:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
                'variables': [{'id': '',
                               'name': self.rssmif16d_variable,
                               'fullPath': self.rssmif16d_variable}]}],
            'stagingLocation': 's3://example-bucket/',
            'subset': {'bbox': bounding_box,
                       'shape': {'href': shape_file_url,
                                 'type': 'application/geo+json'}},
            'user': 'aworden',
        }
        message = Message(message_data)

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        # Ensure the shape file in the Harmony message was downloaded (the
        # logic giving the bounding box precedence over the shape file occurs
        # in `pymods/subset.py`, after the shape file has already been
        # downloaded - however, that file will not be used.
        mock_geojson_download.assert_called_once_with(shape_file_url,
                                                      self.tmp_dir,
                                                      logger=subsetter.logger,
                                                      access_token=message_data['accessToken'],
                                                      cfg=subsetter.config)

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should be the required
        # variables.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=None,
                                           cfg=subsetter.config)

        # Because of the `ANY` match for the request data, the requests for
        # dimensions and all variables will look the same.
        mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=ANY,
                                           cfg=subsetter.config)

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )
        # Ensure the constraint expression contains all the required variables.
        # The index ranges correspond to the bounding box specified in the
        # Harmony message, not the GeoJSON shape file.
        # /wind_speed[][540:599][1320:1379], /time, /longitude[1320:1379]
        # /latitude[540:599]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {'%2Ftime',
             '%2Flatitude%5B540%3A599%5D',
             '%2Flongitude%5B1320%3A1379%5D',
             '%2Fwind_speed%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D'}
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            'opendap_url_wind_speed_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('pymods.dimension_utilities.get_fill_slice')
    @patch('pymods.utilities.uuid4')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.utilities.util_download')
    @patch('harmony.util.stage')
    def test_geo_dimensions(self, mock_stage, mock_util_download, mock_rmtree,
                            mock_mkdtemp, mock_uuid, mock_get_fill_slice):
        """ Ensure a request with explicitly specified dimension extents will
            be correctly processed, requesting only the expected variables,
            with index ranges corresponding to the extents specified.

            To minimise test data in the repository, this test uses geographic
            dimension of latitude and longitude, but within the
            `subset.dimensions` region of the inbound Harmony message.
        """
        bounding_box = [-30, 45, -15, 60]

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path,
                                          all_variables_path]

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2020-01-01T00:00:00.000Z',
                        'end': '2020-01-02T00:00:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
                'variables': [{'id': '',
                               'name': self.rssmif16d_variable,
                               'fullPath': self.rssmif16d_variable}]}],
            'stagingLocation': 's3://example-bucket/',
            'subset': {'dimensions': [
                {'name': 'latitude', 'min': 45, 'max': 60},
                {'name': 'longitude', 'min': 15, 'max': 30}
            ]},
            'user': 'blightyear',
        }
        message = Message(message_data)

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should fetch a NetCDF-4
        # file containing the full 1-D dimension variables only, and the third
        # should retrieve the final NetCDF-4 with all required variables.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_any_call(f'{self.granule_url}.dmr.xml',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=None,
                                           cfg=subsetter.config)

        # Because of the `ANY` match for the request data, the requests for
        # dimensions and all variables will look the same.
        mock_util_download.assert_any_call(f'{self.granule_url}.dap.nc4',
                                           self.tmp_dir,
                                           subsetter.logger,
                                           access_token=message_data['accessToken'],
                                           data=ANY,
                                           cfg=subsetter.config)

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )
        # Ensure the constraint expression contains all the required variables.
        # /wind_speed[][540:599][60:119], /time, /longitude[60:119]
        # /latitude[540:599]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {'%2Ftime',
             '%2Flatitude%5B540%3A599%5D',
             '%2Flongitude%5B60%3A119%5D',
             '%2Fwind_speed%5B%5D%5B540%3A599%5D%5B60%3A119%5D'}
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            'opendap_url_wind_speed_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.subset.download_url')
    @patch('harmony.util.stage')
    def test_exception_handling(self, mock_stage, mock_download_subset,
                                mock_rmtree, mock_mkdtemp):
        """ Ensure that if an exception is raised during processing, this
            causes a HarmonyException to be raised, to allow for informative
            logging.

        """
        mock_mkdtemp.return_value = self.tmp_dir
        mock_download_subset.side_effect = Exception('Random error')

        message_data = {
            'accessToken': 'fake-token',
            'callback': 'https://example.com/',
            'sources': [{
                'granules': [{
                    'id': 'G000-TEST',
                    'url': self.granule_url,
                    'temporal': {
                        'start': '2020-01-01T00:00:00.000Z',
                        'end': '2020-01-02T00:00:00.000Z'
                    },
                    'bbox': [-180, -90, 180, 90]
                }],
                'variables': [{'id': '',
                               'name': self.atl03_variable,
                               'fullPath': self.atl03_variable}]}],
            'stagingLocation': 's3://example-bucket/',
            'user': 'kmattingly',
        }
        message = Message(message_data)

        with self.assertRaises(HarmonyException):
            subsetter = HarmonyAdapter(message, config=config(False))
            subsetter.invoke()

        mock_stage.assert_not_called()
        mock_rmtree.assert_called_once_with(self.tmp_dir)
