from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import patch
import json
import os

import  harmony

from subsetter import HarmonyAdapter
from tests.utilities import (contains, generate_pydap_response, mock_variables,
                             write_dmr)


environment_variables = {'EDL_USERNAME': 'fhaise', 'EDL_PASSWORD': 'A13'}


class TestSubsetterEndToEnd(TestCase):

    @classmethod
    def setUpClass(cls):
        """ Test fixture that can be set once for all tests in the class. """
        cls.granule_url = 'https://harmony.uat.earthdata.nasa.gov/opendap_url'
        cls.variable_full_path = '/gt1r/geophys_corr/geoid'
        cls.expected_variables = ['gt1r_geolocation_delta_time',
                                  'gt1r_geolocation_reference_photon_lon',
                                  'gt1r_geolocation_podppd_flag',
                                  'gt1r_geophys_corr_delta_time',
                                  'gt1r_geolocation_reference_photon_lat',
                                  'gt1r_geophys_corr_geoid']

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

    @patch('pymods.subset.VAR_INFO_SOURCE', 'pydap')
    @patch.object(harmony.BaseHarmonyAdapter, 'completed_with_local_file')
    @patch.object(harmony.BaseHarmonyAdapter, 'cleanup')
    @patch('pymods.subset.mkdtemp')
    @patch('pymods.subset.download_url')
    @patch('pymods.var_info.open_url')
    def test_pydap_end_to_end(self, mock_open_url, mock_download_subset,
                              mock_mkdtemp, mock_cleanup,
                              mock_completed_with_local_file):
        """ Ensure the subsetter will run end-to-end, only mocking the
            HTTP response, and the output interactions with Harmony.

        """
        mock_mkdtemp.return_value = self.tmp_dir
        mock_open_url.return_value = generate_pydap_response(
            mock_variables, {'HDF5_GLOBAL': {'short_name': 'ATL03'}}
        )
        mock_download_subset.return_value = 'opendap_url_subset.nc4'
        expected_variables = ['ancillary_one', 'dimension_one', 'latitude',
                              'longitude', 'science_variable', 'subset_one']

        message_data = {
            'sources': [
                {'granules' : [{'url': self.granule_url}],
                 'variables': [{'id': '',
                                'name': '/science_variable',
                                'fullPath': '/science_variable'}]}
            ],
            'user': 'fhaise'
        }
        message = harmony.message.Message(json.dumps(message_data))

        subsetter = HarmonyAdapter(message)
        granule = subsetter.message.granules[0]
        subsetter.invoke()

        mock_mkdtemp.assert_called_once()
        mock_open_url.assert_called_once_with(self.granule_url)

        mock_download_subset.assert_called_once_with(contains(self.granule_url),
                                                     self.tmp_dir,
                                                     subsetter.logger,
                                                     data='')

        subset_url = mock_download_subset.call_args[0][0]
        self.assertTrue(subset_url.startswith(f'{self.granule_url}.nc4?'))

        requested_variables = subset_url.split('?')[1].split(',')
        self.assertCountEqual(requested_variables, expected_variables)

        mock_completed_with_local_file.assert_called_once_with(
            contains('opendap_url_subset.nc4'),
            source_granule=granule,
            is_regridded=False,
            is_subsetted=False,
            is_variable_subset=True,
            mime='application/octet-stream'
        )

        mock_cleanup.assert_called_once()
