from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import patch
from urllib.parse import parse_qsl
import json

from harmony.message import Message
from harmony.util import config

from subsetter import HarmonyAdapter
from tests.utilities import contains, write_dmr


environment_variables = {'EDL_USERNAME': 'fhaise', 'EDL_PASSWORD': 'A13'}


class TestSubsetterEndToEnd(TestCase):

    @classmethod
    def setUpClass(cls):
        """ Test fixture that can be set once for all tests in the class. """
        cls.granule_url = 'https://harmony.uat.earthdata.nasa.gov/opendap_url'
        cls.variable_full_path = '/gt1r/geophys_corr/geoid'
        cls.expected_variables = ['/gt1r/geolocation/delta_time',
                                  '/gt1r/geolocation/reference_photon_lon',
                                  '/gt1r/geolocation/podppd_flag',
                                  '/gt1r/geophys_corr/delta_time',
                                  '/gt1r/geolocation/reference_photon_lat',
                                  '/gt1r/geophys_corr/geoid']

        with open('tests/data/ATL03_example.dmr', 'r') as file_handler:
            cls.atl03_dmr = file_handler.read()

    def setUp(self):
        """ Have to mock mkdtemp, to know where to put mock .dmr content. """
        self.tmp_dir = mkdtemp()
        self.config = config(validate=False)

    def tearDown(self):
        rmtree(self.tmp_dir)

    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.subset.download_url')
    @patch('pymods.var_info.download_url')
    @patch('harmony.util.stage')
    def test_dmr_end_to_end(self, mock_stage, mock_download_dmr, mock_download_subset,
                            mock_rmtree, mock_mkdtemp):
        """ Ensure the subsetter will run end-to-end, only mocking the
            HTTP response, and the output interactions with Harmony.

        """
        mock_mkdtemp.return_value = self.tmp_dir
        mock_download_dmr.side_effect = [write_dmr(self.tmp_dir, self.atl03_dmr)]
        mock_download_subset.return_value = 'opendap_url_subset.nc4'

        message_data = {
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
                               'name': self.variable_full_path,
                               'fullPath': self.variable_full_path}]}
            ],
            'callback': 'https://example.com/',
            'stagingLocation': 's3://example-bucket/',
            'user': 'fhaise',
            'accessToken': 'fake-token',
        }
        message = Message(json.dumps(message_data))

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        mock_mkdtemp.assert_called_once()
        mock_download_dmr.assert_called_once_with(f'{self.granule_url}.dmr',
                                                  self.tmp_dir,
                                                  subsetter.logger,
                                                  message_data['accessToken'],
                                                  subsetter.config)

        mock_download_subset.assert_called_once_with(contains(self.granule_url),
                                                     self.tmp_dir,
                                                     subsetter.logger,
                                                     access_token=message_data['accessToken'],
                                                     config=subsetter.config)

        subset_url = mock_download_subset.call_args[0][0]
        self.assertTrue(subset_url.startswith(f'{self.granule_url}.dap.nc4?'))

        query_parameters = parse_qsl(subset_url.split('?')[1])
        self.assertEqual('dap4.ce', query_parameters[0][0])

        requested_variables = query_parameters[0][1].split(';')
        self.assertCountEqual(requested_variables, self.expected_variables)

        mock_stage.assert_called_once_with(
            'opendap_url_subset.nc4',
            'opendap_url__gt1r_geophys_corr_geoid.',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger)
