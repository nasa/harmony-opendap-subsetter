from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import ANY, Mock, patch
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

    @patch('pymods.utilities.uuid4')
    @patch('pymods.utilities.copy')
    @patch('subsetter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('pymods.utilities.download_url')
    @patch('pymods.subset.download_url')
    @patch('harmony.util.stage')
    def test_non_geo_end_to_end(self, mock_stage, mock_download_dmr,
                                mock_download_data, mock_rmtree, mock_mkdtemp,
                                mock_copy, mock_uuid):
        """ Ensure the subsetter will run end-to-end, only mocking the
            HTTP responses, and the output interactions with Harmony.

        """
        mock_uuid.return_value = Mock(hex='uuid')
        mock_mkdtemp.return_value = self.tmp_dir
        dmr_path = write_dmr(self.tmp_dir, self.atl03_dmr)
        mock_download_dmr.return_value = dmr_path
        mock_download_data.return_value = 'opendap_url_subset.nc4'
        mock_copy.return_value = 'moved_url_subset.nc4'

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
            'accessToken': None
        }
        message = Message(json.dumps(message_data))

        subsetter = HarmonyAdapter(message, config=config(False))
        subsetter.invoke()

        mock_mkdtemp.assert_called_once()

        mock_download_dmr.assert_called_once_with(
            f'{self.granule_url}.dmr',
            self.tmp_dir,
            subsetter.logger,
            access_token=message_data['accessToken'],
            config=subsetter.config
        )
        mock_download_data.assert_called_once_with(
            f'{self.granule_url}.dap.nc4',
            self.tmp_dir,
            subsetter.logger,
            access_token=message_data['accessToken'],
            config=subsetter.config,
            data=ANY
        )
        mock_copy.assert_called_once_with('opendap_url_subset.nc4',
                                          f'{self.tmp_dir}/uuid.nc4')

        post_data = mock_download_data.call_args[1].get('data', {})
        self.assertIn('dap4.ce', post_data)

        decoded_constraint_expression = unquote(post_data['dap4.ce'])
        requested_variables = decoded_constraint_expression.split(';')
        self.assertCountEqual(requested_variables, self.expected_variables)

        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid.nc4',
            'opendap_url__gt1r_geophys_corr_geoid.',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=subsetter.logger)

    def test_geo_end_to_end(self):
        """ A placeholder test for DAS-1084, in which an end-to-end test should
            be placed, ensuring a full run of the new HOSS functionality is
            successful.

        """
        pass

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
