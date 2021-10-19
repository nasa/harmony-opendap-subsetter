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
        # variables.
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
            'opendap_url_gt1r_geophys_corr_geoid.nc4',
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
        # geographic variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(dimensions_data,
                                       {'%2Flatitude', '%2Flongitude'})
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
        mock_stage.assert_called_once_with(f'{self.tmp_dir}/uuid2.nc4',
                                           'opendap_url_wind_speed.nc4',
                                           'application/x-netcdf4',
                                           location='s3://example-bucket/',
                                           logger=subsetter.logger)
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
        # geographic variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(dimensions_data,
                                       {'%2Flatitude', '%2Flongitude'})
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
        mock_stage.assert_called_once_with(f'{self.tmp_dir}/uuid2.nc4',
                                           'opendap_url_wind_speed.nc4',
                                           'application/x-netcdf4',
                                           location='s3://example-bucket/',
                                           logger=subsetter.logger)
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
        # geographic variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(dimensions_data,
                                       {'%2Flatitude', '%2Flongitude'})
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
        mock_stage.assert_called_once_with(f'{self.tmp_dir}/uuid2.nc4',
                                           'opendap_url_wind_speed.nc4',
                                           'application/x-netcdf4',
                                           location='s3://example-bucket/',
                                           logger=subsetter.logger)
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
        # the first should be the `.dmr`. The second should be the required
        # dimension variables.
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
        # geographic variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(dimensions_data,
                                       {'%2Flatitude', '%2Flongitude'})
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
                                           'opendap_url.nc4',
                                           'application/x-netcdf4',
                                           location='s3://example-bucket/',
                                           logger=subsetter.logger)
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled:
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
