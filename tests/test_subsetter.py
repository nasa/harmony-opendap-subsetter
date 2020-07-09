from unittest import TestCase
from unittest.mock import patch
import json
import os

import  harmony

from subsetter import HarmonyAdapter
from tests.utilities import contains, MockResponse


environment_variables = {'EDL_USERNAME': 'fhaise', 'EDL_PASSWORD': 'A13'}


class TestSubsetterEndToEnd(TestCase):

    @classmethod
    def setUpClass(cls):
        """ Test fixture that can be set once for all tests in the class. """
        cls.granule_url = 'https://harmony.uat.earthdata.nasa.gov/opendap_url'
        cls.variable_name = 'gt1r_geophys_corr_geoid'
        cls.variable_full_path = 'gt1r/geophys_corr/geoid'
        cls.expected_variables = ['gt1r_geolocation_delta_time',
                                  'gt1r_geolocation_reference_photon_lon',
                                  'gt1r_geolocation_podppd_flag',
                                  'gt1r_geophys_corr_delta_time',
                                  'gt1r_geolocation_reference_photon_lat',
                                  'gt1r_geophys_corr_geoid']

        with open('tests/data/ATL03_example.dmr', 'r') as file_handler:
            cls.atl03_dmr = file_handler.read()

    @patch.dict(os.environ, environment_variables)
    @patch.object(harmony.BaseHarmonyAdapter, 'completed_with_local_file')
    @patch.object(harmony.BaseHarmonyAdapter, 'cleanup')
    @patch('requests.get')
    def test_dmr_end_to_end(self, mock_get, mock_cleanup,
                            mock_completed_with_local_file):
        """ Ensure the subsetter will run end-to-end, only mocking the
            HTTP response, and the output interactions with Harmony.

        """
        dmr_response = MockResponse(200, self.atl03_dmr)
        data_response = MockResponse(200, b'Fake binary content')
        mock_get.side_effect = [dmr_response, data_response]

        message_data = {
            'sources': [
                {'granules' : [{'url': self.granule_url}],
                 'variables': [{'id': '',
                                'name': self.variable_name,
                                'fullPath': self.variable_full_path}]}
            ],
            'user': 'fhaise'
        }
        message = harmony.message.Message(json.dumps(message_data))

        reprojector = HarmonyAdapter(message)
        granule = reprojector.message.granules[0]
        reprojector.invoke()

        self.assertEqual(mock_get.call_count, 2)
        mock_get.assert_any_call(f'{self.granule_url}.dmr')

        # The most recent call the requests.get should be to retrieve the
        # subsetted file.
        subset_url = mock_get.call_args[0][0]
        self.assertTrue(subset_url.startswith(f'{self.granule_url}.nc4?'))

        requested_variables = subset_url.split('?')[1].split(',')
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
