from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import patch
from urllib.parse import parse_qsl
import json

import  harmony

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

    def tearDown(self):
        rmtree(self.tmp_dir)

    @patch.object(harmony.BaseHarmonyAdapter, 'completed_with_local_file')
    @patch.object(harmony.BaseHarmonyAdapter, 'cleanup')
    @patch('pymods.subset.mkdtemp')
    @patch('pymods.subset.download_url')
    @patch('pymods.var_info.download_url')
    def test_dmr_end_to_end(self, mock_download_dmr, mock_download_subset,
                            mock_mkdtemp, mock_cleanup,
                            mock_completed_with_local_file):
        """ Ensure the subsetter will run end-to-end, only mocking the
            HTTP response, and the output interactions with Harmony.

        """
        mock_mkdtemp.return_value = self.tmp_dir
        mock_download_dmr.side_effect = [write_dmr(self.tmp_dir, self.atl03_dmr)]
        mock_download_subset.return_value = 'opendap_url_subset.nc4'

        message_data = {
            'sources': [
                {'granules' : [{'url': self.granule_url}],
                 'variables': [{'id': '',
                                'name': self.variable_full_path,
                                'fullPath': self.variable_full_path}]}
            ],
            'user': 'fhaise'
        }
        message = harmony.message.Message(json.dumps(message_data))

        subsetter = HarmonyAdapter(message)
        granule = subsetter.message.granules[0]
        subsetter.invoke()

        mock_mkdtemp.assert_called_once()
        mock_download_dmr.assert_called_once_with(f'{self.granule_url}.dmr',
                                                  self.tmp_dir,
                                                  subsetter.logger)


        mock_download_subset.assert_called_once_with(contains(self.granule_url),
                                                     self.tmp_dir,
                                                     subsetter.logger,
                                                     data='')

        subset_url = mock_download_subset.call_args[0][0]
        self.assertTrue(subset_url.startswith(f'{self.granule_url}.dap.nc4?'))

        query_parameters = parse_qsl(subset_url.split('?')[1])
        self.assertEqual('dap4.ce', query_parameters[0][0])

        requested_variables = query_parameters[0][1].split(';')
        self.assertCountEqual(requested_variables, self.expected_variables)

        mock_completed_with_local_file.assert_called_once_with(
            contains('opendap_url_subset.nc4'),
            source_granule=granule,
            is_regridded=False,
            is_subsetted=False,
            is_variable_subset=True,
            mime='application/octet-stream'
        )

        mock_cleanup.assert_called_once()
