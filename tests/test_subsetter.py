from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import ANY, patch
from urllib.parse import unquote
import json

from harmony.message import Message
from harmony.util import config, HarmonyException

from subsetter import HarmonyAdapter
from tests.utilities import write_dmr


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
    @patch('harmony.util.stage')
    def test_dmr_end_to_end(self, mock_stage, mock_download_subset,
                            mock_rmtree, mock_mkdtemp):
        """ Ensure the subsetter will run end-to-end, only mocking the
            HTTP responses, and the output interactions with Harmony.

        """
        mock_mkdtemp.return_value = self.tmp_dir
        dmr_path = write_dmr(self.tmp_dir, self.atl03_dmr)
        mock_download_subset.side_effect = [dmr_path, 'opendap_url_subset.nc4']

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
                               'fullPath': self.variable_full_path}]}],
            'callback': 'https://example.com/',
            'stagingLocation': 's3://example-bucket/',
            'user': 'fhaise',
            'accessToken': 'fake-token',
        }
        message = Message(json.dumps(message_data))

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        mock_mkdtemp.assert_called_once()

        self.assertEqual(mock_download_subset.call_count, 2)
        mock_download_subset.assert_any_call(f'{self.granule_url}.dmr',
                                             self.tmp_dir,
                                             subsetter.logger,
                                             access_token=message_data['accessToken'],
                                             config=subsetter.config)

        mock_download_subset.assert_any_call(f'{self.granule_url}.dap.nc4',
                                             self.tmp_dir,
                                             subsetter.logger,
                                             access_token=message_data['accessToken'],
                                             config=subsetter.config,
                                             data=ANY)

        post_data = mock_download_subset.call_args[1].get('data', {})
        self.assertIn('dap4.ce', post_data)

        decoded_constraint_expression = unquote(post_data['dap4.ce'])
        requested_variables = decoded_constraint_expression.split(';')
        self.assertCountEqual(requested_variables, self.expected_variables)

        mock_stage.assert_called_once_with(
            'opendap_url_subset.nc4',
            'opendap_url__gt1r_geophys_corr_geoid.',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger)

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
        dmr_path = write_dmr(self.tmp_dir, self.atl03_dmr)
        mock_download_subset.side_effect = Exception('Random error')

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
                               'fullPath': self.variable_full_path}]}],
            'callback': 'https://example.com/',
            'stagingLocation': 's3://example-bucket/',
            'user': 'fhaise',
            'accessToken': 'fake-token',
        }
        message = Message(json.dumps(message_data))

        with self.assertRaises(HarmonyException):
            subsetter = HarmonyAdapter(message, config=config(False))
            subsetter.invoke()
