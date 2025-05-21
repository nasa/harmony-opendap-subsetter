import shutil
from logging import Logger
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import call, patch

import numpy as np
from harmony.message import Message, Source
from harmony.message import Variable as HarmonyVariable
from harmony.util import config
from netCDF4 import Dataset
from varinfo import VarInfoFromDmr

from hoss.subset import (
    fill_variable,
    fill_variables,
    get_required_variables,
    get_varinfo,
    subset_granule,
)


class TestSubset(TestCase):
    """Test the module that performs subsetting on a single granule."""

    @classmethod
    def setUpClass(cls):
        """Define test assets that can be shared between tests."""
        cls.access_token = 'access'
        cls.bounding_box = [40, -30, 50, -20]
        cls.config = config(validate=False)
        cls.collection_short_name = 'RSSMIF16D'
        cls.granule_url = 'https://harmony.earthdata.nasa.gov/bucket/rssmif16d'
        cls.logger = Logger('tests')
        cls.output_path = 'f16_ssmis_subset.nc4'
        cls.required_variables = {'/latitude', '/longitude', '/time', '/rainfall_rate'}
        cls.harmony_source = Source(
            {
                'collection': 'C1234567890-PROV',
                'shortName': cls.collection_short_name,
                'variables': [
                    {
                        'id': 'V1238395077-EEDTEST',
                        'name': '/rainfall_rate',
                        'fullPath': '/rainfall_rate',
                    }
                ],
            }
        )
        cls.varinfo = VarInfoFromDmr('tests/data/rssmif16d_example.dmr')

    def setUp(self):
        """Define test assets that should not be shared between tests."""
        self.output_dir = mkdtemp()

    def tearDown(self):
        """Clean-up to perform between every test."""
        shutil.rmtree(self.output_dir)

    @patch('hoss.subset.fill_variables')
    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_requested_index_ranges')
    @patch('hoss.subset.get_temporal_index_ranges')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_granule_not_geo(
        self,
        mock_get_varinfo,
        mock_prefetch_dimensions,
        mock_get_spatial_index_ranges,
        mock_get_temporal_index_ranges,
        mock_get_requested_index_ranges,
        mock_get_opendap_nc4,
        mock_fill_variables,
    ):
        """Ensure a request to extract only a variable subset runs without
        error. Because no bounding box and no temporal range is specified
        in this request, the prefetch dimension utility functionality, the
        HOSS functionality in `hoss.spatial.py`  and the functionality in
        `hoss.temporal.py` should not be called.

        """
        harmony_message = Message({'accessToken': self.access_token})
        mock_get_varinfo.return_value = self.varinfo
        mock_get_opendap_nc4.return_value = self.output_path

        output_path = subset_granule(
            self.granule_url,
            self.harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertEqual(output_path, self.output_path)

        mock_get_varinfo.assert_called_once_with(
            self.granule_url,
            self.output_dir,
            self.logger,
            self.collection_short_name,
            self.access_token,
            self.config,
        )
        mock_get_opendap_nc4.assert_called_once_with(
            self.granule_url,
            self.required_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )
        mock_fill_variables.assert_called_once_with(
            self.output_path, self.varinfo, self.required_variables, {}
        )

        mock_prefetch_dimensions.assert_not_called()
        mock_get_spatial_index_ranges.assert_not_called()
        mock_get_temporal_index_ranges.assert_not_called()
        mock_get_requested_index_ranges.assert_not_called()

    @patch('hoss.subset.fill_variables')
    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_requested_index_ranges')
    @patch('hoss.subset.get_temporal_index_ranges')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_granule_geo(
        self,
        mock_get_varinfo,
        mock_prefetch_dimensions,
        mock_get_spatial_index_ranges,
        mock_get_temporal_index_ranges,
        mock_get_requested_index_ranges,
        mock_get_opendap_nc4,
        mock_fill_variables,
    ):
        """Ensure a request to extract both a variable and spatial subset runs
        without error. Because a bounding box is specified in this request,
        the prefetch dimension utility functionality and the HOSS
        functionality in `hoss.spatial.py` should be called. However,
        because there is no specified `temporal_range`, the functionality
        in `hoss.temporal.py` should not be called.

        """
        harmony_message = Message(
            {'accessToken': self.access_token, 'subset': {'bbox': self.bounding_box}}
        )
        index_ranges = {'/latitude': (240, 279), '/longitude': (160, 199)}
        prefetch_path = 'prefetch.nc4'
        variables_with_ranges = {
            '/latitude[240:279]',
            '/longitude[160:199]',
            '/rainfall_rate[][240:279][160:199]',
            '/time',
        }

        mock_get_varinfo.return_value = self.varinfo
        mock_prefetch_dimensions.return_value = prefetch_path
        mock_get_spatial_index_ranges.return_value = index_ranges
        mock_get_opendap_nc4.return_value = self.output_path

        output_path = subset_granule(
            self.granule_url,
            self.harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertEqual(output_path, self.output_path)
        mock_get_varinfo.assert_called_once_with(
            self.granule_url,
            self.output_dir,
            self.logger,
            self.collection_short_name,
            self.access_token,
            self.config,
        )

        mock_prefetch_dimensions.assert_called_once_with(
            self.granule_url,
            self.varinfo,
            self.required_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_get_temporal_index_ranges.assert_not_called()
        mock_get_spatial_index_ranges.assert_called_once_with(
            self.required_variables, self.varinfo, prefetch_path, harmony_message, None
        )

        mock_get_requested_index_ranges.assert_not_called()
        mock_get_opendap_nc4.assert_called_once_with(
            self.granule_url,
            variables_with_ranges,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_fill_variables.assert_called_once_with(
            self.output_path, self.varinfo, self.required_variables, index_ranges
        )

    @patch('hoss.subset.fill_variables')
    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_requested_index_ranges')
    @patch('hoss.subset.get_temporal_index_ranges')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_non_geo_no_variables(
        self,
        mock_get_varinfo,
        mock_prefetch_dimensions,
        mock_get_spatial_index_ranges,
        mock_get_temporal_index_ranges,
        mock_get_requested_index_ranges,
        mock_get_opendap_nc4,
        mock_fill_variables,
    ):
        """Ensure a request without a bounding box and without any specified
        variables will produce a request to OPeNDAP that does not specify
        any variables. This will default to retrieving the full NetCDF-4
        file from OPeNDAP. The prefetch dimension functionality and the
        HOSS functionality in both `hoss.spatial.py` and
        `hoss.temporal.py` should not be called.

        """
        harmony_source = Source(
            {
                'collection': 'C1234567890-EEDTEST',
                'shortName': self.collection_short_name,
            }
        )
        harmony_message = Message({'accessToken': self.access_token})
        expected_variables = set()
        index_ranges = {}
        mock_get_varinfo.return_value = self.varinfo
        mock_get_opendap_nc4.return_value = self.output_path

        output_path = subset_granule(
            self.granule_url,
            harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertEqual(output_path, self.output_path)
        mock_get_varinfo.assert_called_once_with(
            self.granule_url,
            self.output_dir,
            self.logger,
            self.collection_short_name,
            self.access_token,
            self.config,
        )

        mock_get_opendap_nc4.assert_called_once_with(
            self.granule_url,
            expected_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_fill_variables.assert_called_once_with(
            self.output_path, self.varinfo, expected_variables, index_ranges
        )

        mock_prefetch_dimensions.assert_not_called()
        mock_get_spatial_index_ranges.assert_not_called()
        mock_get_temporal_index_ranges.assert_not_called()
        mock_get_requested_index_ranges.assert_not_called()

    @patch('hoss.subset.fill_variables')
    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_requested_index_ranges')
    @patch('hoss.subset.get_temporal_index_ranges')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_geo_no_variables(
        self,
        mock_get_varinfo,
        mock_prefetch_dimensions,
        mock_get_spatial_index_ranges,
        mock_get_temporal_index_ranges,
        mock_get_requested_index_ranges,
        mock_get_opendap_nc4,
        mock_fill_variables,
    ):
        """Ensure a request with a bounding box, but without any specified
        variables will consider all science and metadata variables as the
        requested variables. This situation will arise if a user requests
        all variables. HOSS will need to explicitly list all the variables
        it retrieves as the DAP4 constraint expression will need to specify
        index ranges for all geographically gridded variables. Both the
        prefetch dimension functionality and the HOSS functionality in
        `hoss.spatial.py` should be called. However, because there is no
        specified `temporal_range`, the functionality in `hoss.temporal.py`
        should not be called.

        """
        harmony_source = Source(
            {
                'collection': 'C1234567890-EEDTEST',
                'shortName': self.collection_short_name,
            }
        )
        harmony_message = Message(
            {'accessToken': self.access_token, 'subset': {'bbox': self.bounding_box}}
        )
        expected_variables = {
            '/atmosphere_cloud_liquid_water_content',
            '/atmosphere_water_vapor_content',
            '/latitude',
            '/longitude',
            '/rainfall_rate',
            '/sst_dtime',
            '/time',
            '/wind_speed',
        }
        index_ranges = {'/latitude': (240, 279), '/longitude': (160, 199)}
        prefetch_path = 'prefetch.nc4'
        variables_with_ranges = {
            '/atmosphere_cloud_liquid_water_content[][240:279][160:199]',
            '/atmosphere_water_vapor_content[][240:279][160:199]',
            '/latitude[240:279]',
            '/longitude[160:199]',
            '/rainfall_rate[][240:279][160:199]',
            '/sst_dtime[][240:279][160:199]',
            '/time',
            '/wind_speed[][240:279][160:199]',
        }

        mock_get_varinfo.return_value = self.varinfo
        mock_prefetch_dimensions.return_value = prefetch_path
        mock_get_spatial_index_ranges.return_value = index_ranges
        mock_get_opendap_nc4.return_value = self.output_path

        output_path = subset_granule(
            self.granule_url,
            harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertEqual(output_path, self.output_path)
        mock_get_varinfo.assert_called_once_with(
            self.granule_url,
            self.output_dir,
            self.logger,
            self.collection_short_name,
            self.access_token,
            self.config,
        )

        mock_prefetch_dimensions.assert_called_once_with(
            self.granule_url,
            self.varinfo,
            expected_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_get_temporal_index_ranges.assert_not_called()
        mock_get_spatial_index_ranges.assert_called_once_with(
            expected_variables, self.varinfo, prefetch_path, harmony_message, None
        )

        mock_get_requested_index_ranges.assert_not_called()
        mock_get_opendap_nc4.assert_called_once_with(
            self.granule_url,
            variables_with_ranges,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_fill_variables.assert_called_once_with(
            self.output_path, self.varinfo, expected_variables, index_ranges
        )

    @patch('hoss.subset.fill_variables')
    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_requested_index_ranges')
    @patch('hoss.subset.get_temporal_index_ranges')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_non_variable_dimensions(
        self,
        mock_get_varinfo,
        mock_prefetch_dimensions,
        mock_get_spatial_index_ranges,
        mock_get_temporal_index_ranges,
        mock_get_requested_index_ranges,
        mock_get_opendap_nc4,
        mock_fill_variables,
    ):
        """Ensure a request with a bounding box, without specified variables,
        will not include non-variable dimensions in the DAP4 constraint
        expression of the final request to OPeNDAP

        In the GPM_3IMERGHH data, the specific dimensions that should not
        be included in the required variables are `latv`, `lonv` and `nv`.
        These are size-only dimensions for the `lat_bnds`, `lon_bnds` and
        `time_bnds` variables.

        """
        harmony_source = Source(
            {
                'collection': 'C1234567890-EEDTEST',
                'shortName': self.collection_short_name,
            }
        )
        harmony_message = Message(
            {'accessToken': self.access_token, 'subset': {'bbox': self.bounding_box}}
        )
        url = 'https://harmony.earthdata.nasa.gov/bucket/GPM'
        varinfo = VarInfoFromDmr('tests/data/GPM_3IMERGHH_example.dmr')
        expected_variables = {
            '/Grid/HQobservationTime',
            '/Grid/HQprecipitation',
            '/Grid/HQprecipSource',
            '/Grid/IRkalmanFilterWeight',
            '/Grid/IRprecipitation',
            '/Grid/lat',
            '/Grid/lat_bnds',
            '/Grid/lon',
            '/Grid/lon_bnds',
            '/Grid/precipitationCal',
            '/Grid/precipitationQualityIndex',
            '/Grid/precipitationUncal',
            '/Grid/probabilityLiquidPrecipitation',
            '/Grid/randomError',
            '/Grid/time',
            '/Grid/time_bnds',
        }

        prefetch_path = 'GPM_prefetch.nc'
        index_ranges = {'/Grid/lat': (600, 699), '/Grid/lon': (2200, 2299)}
        expected_output_path = 'GPM_3IMERGHH_subset.nc4'

        variables_with_ranges = {
            '/Grid/HQobservationTime[][2200:2299][600:699]',
            '/Grid/HQprecipitation[][2200:2299][600:699]',
            '/Grid/HQprecipSource[][2200:2299][600:699]',
            '/Grid/IRkalmanFilterWeight[][2200:2299][600:699]',
            '/Grid/IRprecipitation[][2200:2299][600:699]',
            '/Grid/lat[600:699]',
            '/Grid/lat_bnds[600:699][]',
            '/Grid/lon[2200:2299]',
            '/Grid/lon_bnds[2200:2299][]',
            '/Grid/precipitationCal[][2200:2299][600:699]',
            '/Grid/precipitationQualityIndex[][2200:2299][600:699]',
            '/Grid/precipitationUncal[][2200:2299][600:699]',
            '/Grid/probabilityLiquidPrecipitation[][2200:2299][600:699]',
            '/Grid/randomError[][2200:2299][600:699]',
            '/Grid/time',
            '/Grid/time_bnds',
        }

        mock_get_varinfo.return_value = varinfo
        mock_prefetch_dimensions.return_value = prefetch_path
        mock_get_spatial_index_ranges.return_value = index_ranges
        mock_get_opendap_nc4.return_value = expected_output_path

        output_path = subset_granule(
            url,
            harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertEqual(output_path, expected_output_path)
        mock_get_varinfo.assert_called_once_with(
            url,
            self.output_dir,
            self.logger,
            self.collection_short_name,
            self.access_token,
            self.config,
        )

        mock_prefetch_dimensions.assert_called_once_with(
            url,
            varinfo,
            expected_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_get_temporal_index_ranges.assert_not_called()
        mock_get_spatial_index_ranges.assert_called_once_with(
            expected_variables, varinfo, prefetch_path, harmony_message, None
        )

        mock_get_requested_index_ranges.assert_not_called()
        mock_get_opendap_nc4.assert_called_once_with(
            url,
            variables_with_ranges,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_fill_variables.assert_called_once_with(
            expected_output_path, varinfo, expected_variables, index_ranges
        )

    @patch('hoss.subset.fill_variables')
    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_requested_index_ranges')
    @patch('hoss.subset.get_temporal_index_ranges')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_bounds_reference(
        self,
        mock_get_varinfo,
        mock_prefetch_dimensions,
        mock_get_spatial_index_ranges,
        mock_get_temporal_index_ranges,
        mock_get_requested_index_ranges,
        mock_get_opendap_nc4,
        mock_fill_variables,
    ):
        """Ensure a request with a bounding box, specifying variables that
        have references in a `bounds` attribute also consider the variables
        referred to in the `bounds` attribute as required.

        In the GPM_3IMERGHH data, the `lat`, `lon` and `time` variables
        have `lat_bnds`, `lon_bnds` and `time_bnds`, respectively.

        """
        harmony_source = Source(
            {
                'collection': 'C1234567890-EEDTEST',
                'shortName': self.collection_short_name,
                'variables': [
                    {'fullPath': '/Grid/lon', 'id': 'V123-EEDTEST', 'name': '/Grid/lon'}
                ],
            }
        )
        harmony_message = Message(
            {'accessToken': self.access_token, 'subset': {'bbox': self.bounding_box}}
        )
        url = 'https://harmony.earthdata.nasa.gov/bucket/GPM'
        varinfo = VarInfoFromDmr('tests/data/GPM_3IMERGHH_example.dmr')

        expected_variables = {'/Grid/lon', '/Grid/lon_bnds'}

        prefetch_path = 'GPM_prefetch.nc'
        index_ranges = {'/Grid/lon': (2200, 2299)}
        expected_output_path = 'GPM_3IMERGHH_subset.nc4'

        variables_with_ranges = {'/Grid/lon[2200:2299]', '/Grid/lon_bnds[2200:2299][]'}

        mock_get_varinfo.return_value = varinfo
        mock_prefetch_dimensions.return_value = prefetch_path
        mock_get_spatial_index_ranges.return_value = index_ranges
        mock_get_opendap_nc4.return_value = expected_output_path

        output_path = subset_granule(
            url,
            harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertIn('GPM_3IMERGHH_subset.nc4', output_path)
        mock_get_varinfo.assert_called_once_with(
            url,
            self.output_dir,
            self.logger,
            self.collection_short_name,
            self.access_token,
            self.config,
        )

        mock_prefetch_dimensions.assert_called_once_with(
            url,
            varinfo,
            expected_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_get_temporal_index_ranges.assert_not_called()
        mock_get_spatial_index_ranges.assert_called_once_with(
            expected_variables, varinfo, prefetch_path, harmony_message, None
        )

        mock_get_requested_index_ranges.assert_not_called()
        mock_get_opendap_nc4.assert_called_once_with(
            url,
            variables_with_ranges,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_fill_variables.assert_called_once_with(
            expected_output_path, varinfo, expected_variables, index_ranges
        )

    @patch('hoss.subset.fill_variables')
    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_requested_index_ranges')
    @patch('hoss.subset.get_temporal_index_ranges')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_temporal(
        self,
        mock_get_varinfo,
        mock_prefetch_dimensions,
        mock_get_spatial_index_ranges,
        mock_get_temporal_index_ranges,
        mock_get_requested_index_ranges,
        mock_get_opendap_nc4,
        mock_fill_variables,
    ):
        """Ensure a request with a temporal range constructs an OPeNDAP
        request that contains index range values for only the temporal
        dimension of the data.

        """
        url = 'https://harmony.earthdata.nasa.gov/bucket/M2T1NXSLV'
        harmony_source = Source(
            {
                'collection': 'C1234567890-PROVIDER',
                'shortName': self.collection_short_name,
                'variables': [{'fullPath': '/PS', 'id': 'V123-EEDTEST', 'name': '/PS'}],
            }
        )
        harmony_message = Message(
            {
                'accessToken': self.access_token,
                'temporal': {
                    'start': '2021-01-10T01:00:00',
                    'end': '2021-01-10T03:00:00',
                },
            }
        )
        varinfo = VarInfoFromDmr(
            'tests/data/M2T1NXSLV_example.dmr', config_file='hoss/hoss_config.json'
        )

        expected_variables = {'/PS', '/lat', '/lon', '/time'}

        prefetch_path = 'M2T1NXSLV_prefetch.nc4'
        index_ranges = {'/time': (1, 2)}
        expected_output_path = 'M2T1NXSLV_subset.nc4'

        variables_with_ranges = {'/PS[1:2][][]', '/lat', '/lon', '/time[1:2]'}

        mock_get_varinfo.return_value = varinfo
        mock_prefetch_dimensions.return_value = prefetch_path
        mock_get_temporal_index_ranges.return_value = index_ranges
        mock_get_opendap_nc4.return_value = expected_output_path

        output_path = subset_granule(
            url,
            harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertIn('M2T1NXSLV_subset.nc4', output_path)
        mock_get_varinfo.assert_called_once_with(
            url,
            self.output_dir,
            self.logger,
            self.collection_short_name,
            self.access_token,
            self.config,
        )

        mock_prefetch_dimensions.assert_called_once_with(
            url,
            varinfo,
            expected_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_get_spatial_index_ranges.assert_not_called()
        mock_get_temporal_index_ranges.assert_called_once_with(
            expected_variables, varinfo, prefetch_path, harmony_message
        )

        mock_get_requested_index_ranges.assert_not_called()
        mock_get_opendap_nc4.assert_called_once_with(
            url,
            variables_with_ranges,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_fill_variables.assert_called_once_with(
            expected_output_path, varinfo, expected_variables, index_ranges
        )

    @patch('hoss.subset.fill_variables')
    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_requested_index_ranges')
    @patch('hoss.subset.get_temporal_index_ranges')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_geo_temporal(
        self,
        mock_get_varinfo,
        mock_prefetch_dimensions,
        mock_get_spatial_index_ranges,
        mock_get_temporal_index_ranges,
        mock_get_requested_index_ranges,
        mock_get_opendap_nc4,
        mock_fill_variables,
    ):
        """Ensure a request with a temporal range and a bounding box
        constructs an OPeNDAP request that contains index range values for
        both the geographic and the temporal dimensions of the data.

        """
        url = 'https://harmony.earthdata.nasa.gov/bucket/M2T1NXSLV'
        harmony_source = Source(
            {
                'collection': 'C1234567890-EEDTEST',
                'shortName': self.collection_short_name,
                'variables': [{'fullPath': '/PS', 'id': 'V123-EEDTEST', 'name': '/PS'}],
            }
        )
        harmony_message = Message(
            {
                'accessToken': self.access_token,
                'subset': {'bbox': self.bounding_box},
                'temporal': {
                    'start': '2021-01-10T01:00:00',
                    'end': '2021-01-10T03:00:00',
                },
            }
        )
        varinfo = VarInfoFromDmr(
            'tests/data/M2T1NXSLV_example.dmr', config_file='hoss/hoss_config.json'
        )

        expected_variables = {'/PS', '/lat', '/lon', '/time'}

        prefetch_path = 'M2T1NXSLV_prefetch.nc4'
        geo_index_ranges = {'/lat': (120, 140), '/lon': (352, 368)}
        temporal_index_ranges = {'/time': (1, 2)}
        all_index_ranges = geo_index_ranges.copy()
        all_index_ranges.update(temporal_index_ranges)
        expected_output_path = 'M2T1NXSLV_subset.nc4'

        variables_with_ranges = {
            '/PS[1:2][120:140][352:368]',
            '/lat[120:140]',
            '/lon[352:368]',
            '/time[1:2]',
        }

        mock_get_varinfo.return_value = varinfo
        mock_prefetch_dimensions.return_value = prefetch_path
        mock_get_temporal_index_ranges.return_value = temporal_index_ranges
        mock_get_spatial_index_ranges.return_value = geo_index_ranges
        mock_get_opendap_nc4.return_value = expected_output_path

        output_path = subset_granule(
            url,
            harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertIn('M2T1NXSLV_subset.nc4', output_path)
        mock_get_varinfo.assert_called_once_with(
            url,
            self.output_dir,
            self.logger,
            self.collection_short_name,
            self.access_token,
            self.config,
        )

        mock_prefetch_dimensions.assert_called_once_with(
            url,
            varinfo,
            expected_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_get_spatial_index_ranges.assert_called_once_with(
            expected_variables, varinfo, prefetch_path, harmony_message, None
        )

        mock_get_temporal_index_ranges.assert_called_once_with(
            expected_variables, varinfo, prefetch_path, harmony_message
        )

        mock_get_requested_index_ranges.assert_not_called()
        mock_get_opendap_nc4.assert_called_once_with(
            url,
            variables_with_ranges,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_fill_variables.assert_called_once_with(
            expected_output_path, varinfo, expected_variables, all_index_ranges
        )

    @patch('hoss.subset.fill_variables')
    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_requested_index_ranges')
    @patch('hoss.subset.get_temporal_index_ranges')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_request_shape_file')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_granule_shape(
        self,
        mock_get_varinfo,
        mock_prefetch_dimensions,
        mock_get_request_shape_file,
        mock_get_spatial_index_ranges,
        mock_get_temporal_index_ranges,
        mock_get_requested_index_ranges,
        mock_get_opendap_nc4,
        mock_fill_variables,
    ):
        """Ensure a request to extract both a variable and spatial subset runs
        without error. This request will have specified a shape file rather
        than a bounding box, which should be passed along to the
        `get_spatial_index_ranges` function. The prefetch dimension utility
        functionality and the HOSS functionality in `hoss.spatial.py`
        should be called. However, because there is no specified
        `temporal_range`, the functionality in `hoss.temporal.py` should
        not be called.

        """
        shape_file_path = 'tests/geojson_examples/polygon.geo.json'
        mock_get_request_shape_file.return_value = shape_file_path
        harmony_message = Message(
            {
                'accessToken': self.access_token,
                'subset': {
                    'shape': {
                        'href': 'https://example.com/polygon.geo.json',
                        'type': 'application/geo+json',
                    }
                },
            }
        )
        index_ranges = {'/latitude': (508, 527), '/longitude': (983, 1003)}
        prefetch_path = 'prefetch.nc4'
        variables_with_ranges = {
            '/latitude[508:527]',
            '/longitude[983:1003]',
            '/rainfall_rate[][508:527][983:1003]',
            '/time',
        }

        mock_get_varinfo.return_value = self.varinfo
        mock_prefetch_dimensions.return_value = prefetch_path
        mock_get_spatial_index_ranges.return_value = index_ranges
        mock_get_opendap_nc4.return_value = self.output_path

        output_path = subset_granule(
            self.granule_url,
            self.harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertEqual(output_path, self.output_path)
        mock_get_varinfo.assert_called_once_with(
            self.granule_url,
            self.output_dir,
            self.logger,
            self.collection_short_name,
            self.access_token,
            self.config,
        )

        mock_prefetch_dimensions.assert_called_once_with(
            self.granule_url,
            self.varinfo,
            self.required_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_get_temporal_index_ranges.assert_not_called()
        mock_get_request_shape_file.assert_called_once_with(
            harmony_message, self.output_dir, self.logger, self.config
        )
        mock_get_spatial_index_ranges.assert_called_once_with(
            self.required_variables,
            self.varinfo,
            prefetch_path,
            harmony_message,
            shape_file_path,
        )

        mock_get_requested_index_ranges.assert_not_called()
        mock_get_opendap_nc4.assert_called_once_with(
            self.granule_url,
            variables_with_ranges,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_fill_variables.assert_called_once_with(
            self.output_path, self.varinfo, self.required_variables, index_ranges
        )

    @patch('hoss.subset.fill_variables')
    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_requested_index_ranges')
    @patch('hoss.subset.get_temporal_index_ranges')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_request_shape_file')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_granule_shape_and_bbox(
        self,
        mock_get_varinfo,
        mock_prefetch_dimensions,
        mock_get_request_shape_file,
        mock_get_spatial_index_ranges,
        mock_get_temporal_index_ranges,
        mock_get_requested_index_ranges,
        mock_get_opendap_nc4,
        mock_fill_variables,
    ):
        """Ensure a request to extract both a variable and spatial subset runs
        without error. This request will have specified both a bounding box
        and a shape file, both of which will be passed along to
        `get_spatial_index_ranges`, so that it can determine which to use.
        The prefetch dimension utility functionality and the HOSS
        functionality in `hoss.spatial.py` should be called. However,
        because there is no specified `temporal_range`, the functionality
        in `hoss.temporal.py` should not be called.

        """
        shape_file_path = 'tests/geojson_examples/polygon.geo.json'
        mock_get_request_shape_file.return_value = shape_file_path
        harmony_message = Message(
            {
                'accessToken': self.access_token,
                'subset': {
                    'bbox': self.bounding_box,
                    'shape': {
                        'href': 'https://example.com/polygon.geo.json',
                        'type': 'application/geo+json',
                    },
                },
            }
        )

        index_ranges = {'/latitude': (240, 279), '/longitude': (160, 199)}
        prefetch_path = 'prefetch.nc4'
        variables_with_ranges = {
            '/latitude[240:279]',
            '/longitude[160:199]',
            '/rainfall_rate[][240:279][160:199]',
            '/time',
        }

        mock_get_varinfo.return_value = self.varinfo
        mock_prefetch_dimensions.return_value = prefetch_path
        mock_get_spatial_index_ranges.return_value = index_ranges
        mock_get_opendap_nc4.return_value = self.output_path

        output_path = subset_granule(
            self.granule_url,
            self.harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertEqual(output_path, self.output_path)
        mock_get_varinfo.assert_called_once_with(
            self.granule_url,
            self.output_dir,
            self.logger,
            self.collection_short_name,
            self.access_token,
            self.config,
        )

        mock_prefetch_dimensions.assert_called_once_with(
            self.granule_url,
            self.varinfo,
            self.required_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_get_temporal_index_ranges.assert_not_called()
        mock_get_request_shape_file.assert_called_once_with(
            harmony_message, self.output_dir, self.logger, self.config
        )
        mock_get_spatial_index_ranges.assert_called_once_with(
            self.required_variables,
            self.varinfo,
            prefetch_path,
            harmony_message,
            shape_file_path,
        )

        mock_get_requested_index_ranges.asset_not_called()
        mock_get_opendap_nc4.assert_called_once_with(
            self.granule_url,
            variables_with_ranges,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_fill_variables.assert_called_once_with(
            self.output_path, self.varinfo, self.required_variables, index_ranges
        )

    @patch('hoss.subset.fill_variables')
    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_requested_index_ranges')
    @patch('hoss.subset.get_temporal_index_ranges')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_granule_geo_named(
        self,
        mock_get_varinfo,
        mock_prefetch_dimensions,
        mock_get_spatial_index_ranges,
        mock_get_temporal_index_ranges,
        mock_get_requested_index_ranges,
        mock_get_opendap_nc4,
        mock_fill_variables,
    ):
        """Ensure a request to extract both a variable and named dimension
        subset runs without error. Because a dimension is specified in this
        request, the prefetch dimension utility functionality and the HOSS
        functionality in `hoss.spatial.py` should be called. However,
        because there is no specified `temporal_range`, the functionality
        in `hoss.temporal.py` should not be called.

        This test will use spatial dimensions, but explicitly naming them
        instead of using a bounding box.

        """
        harmony_message = Message(
            {
                'accessToken': self.access_token,
                'subset': {
                    'dimensions': [
                        {'name': '/latitude', 'min': -30, 'max': -20},
                        {'name': '/longitude', 'min': 40, 'max': 50},
                    ]
                },
            }
        )
        index_ranges = {'/latitude': (240, 279), '/longitude': (160, 199)}
        prefetch_path = 'prefetch.nc4'
        variables_with_ranges = {
            '/latitude[240:279]',
            '/longitude[160:199]',
            '/rainfall_rate[][240:279][160:199]',
            '/time',
        }

        mock_get_varinfo.return_value = self.varinfo
        mock_prefetch_dimensions.return_value = prefetch_path
        mock_get_requested_index_ranges.return_value = index_ranges
        mock_get_opendap_nc4.return_value = self.output_path

        output_path = subset_granule(
            self.granule_url,
            self.harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertEqual(output_path, self.output_path)
        mock_get_varinfo.assert_called_once_with(
            self.granule_url,
            self.output_dir,
            self.logger,
            self.collection_short_name,
            self.access_token,
            self.config,
        )

        mock_prefetch_dimensions.assert_called_once_with(
            self.granule_url,
            self.varinfo,
            self.required_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_get_temporal_index_ranges.assert_not_called()
        mock_get_spatial_index_ranges.assert_not_called()
        mock_get_requested_index_ranges.assert_called_once_with(
            self.required_variables, self.varinfo, prefetch_path, harmony_message
        )

        mock_get_opendap_nc4.assert_called_once_with(
            self.granule_url,
            variables_with_ranges,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

        mock_fill_variables.assert_called_once_with(
            self.output_path, self.varinfo, self.required_variables, index_ranges
        )

    @patch('hoss.subset.download_url')
    def test_get_varinfo(self, mock_download_url):
        """Ensure a request is made to OPeNDAP to retrieve the `.dmr` and
        that a `VarInfoFromDmr` instance can be created from that
        downloaded file.

        """
        dmr_path = shutil.copy(
            'tests/data/rssmif16d_example.dmr',
            f'{self.output_dir}/rssmif16d_example.dmr',
        )
        mock_download_url.return_value = dmr_path

        varinfo = get_varinfo(
            self.granule_url,
            self.output_dir,
            self.logger,
            self.collection_short_name,
            self.access_token,
            self.config,
        )

        self.assertIsInstance(varinfo, VarInfoFromDmr)
        self.assertSetEqual(
            set(varinfo.variables.keys()),
            {
                '/atmosphere_cloud_liquid_water_content',
                '/atmosphere_water_vapor_content',
                '/latitude',
                '/longitude',
                '/rainfall_rate',
                '/sst_dtime',
                '/time',
                '/wind_speed',
            },
        )

    def test_get_required_variables(self):
        """Ensure that all requested variables are extracted from the list of
        variables in the Harmony message. Alternatively, if no variables
        are specified, all variables in the `.dmr` should be returned.

        After the requested variables have been identified, the return
        value should also include all those variables that support those
        requested (e.g., dimensions, coordinates, etc).

        * Test case 1: variables in message - the variable paths should be
                       extracted.
        * Test case 2: variables in message, some without leading slash -
                       the variables paths should be extracted with a
                       slash prepended to each.
        * Test case 3: variables in message, index ranges required (e.g.,
                       for bounding box, shape file or temporal subset)
                       - the same variable paths from the message should be
                       extracted.
        * Test case 4: variables not in message, no index ranges required
                       - the output should be an empty set (straight
                       variable subset, all variables from OPeNDAP).
        * Test case 5: variables not in message, index ranges required
                       (e.g., for bounding box, shape file or temporal
                       subset) - the return value should include all
                       non-dimension variables from the `VarInfoFromDmr`
                       instance.
        * Test case 6: variables not in message, but configured as required
                       in the json file.

        """
        all_variables = {
            '/atmosphere_cloud_liquid_water_content',
            '/atmosphere_water_vapor_content',
            '/latitude',
            '/longitude',
            '/rainfall_rate',
            '/sst_dtime',
            '/time',
            '/wind_speed',
        }

        with self.subTest('Variables specified, no index range subset:'):
            harmony_variables = [
                HarmonyVariable(
                    {
                        'fullPath': '/rainfall_rate',
                        'id': 'V1234-PROVIDER',
                        'name': '/rainfall_rate',
                    }
                )
            ]
            self.assertSetEqual(
                get_required_variables(
                    self.varinfo, harmony_variables, False, self.logger
                ),
                {'/latitude', '/longitude', '/rainfall_rate', '/time'},
            )

        with self.subTest('Variable without leading slash can be handled'):
            harmony_variables = [
                HarmonyVariable(
                    {
                        'fullPath': 'rainfall_rate',
                        'id': 'V1234-PROVIDER',
                        'name': 'rainfall_rate',
                    }
                )
            ]
            self.assertSetEqual(
                get_required_variables(
                    self.varinfo, harmony_variables, False, self.logger
                ),
                {'/latitude', '/longitude', '/rainfall_rate', '/time'},
            )

        with self.subTest('Variables specified for an index_range_subset'):
            harmony_variables = [
                HarmonyVariable(
                    {
                        'fullPath': '/rainfall_rate',
                        'id': 'V1234-PROVIDER',
                        'name': '/rainfall_rate',
                    }
                )
            ]
            self.assertSetEqual(
                get_required_variables(
                    self.varinfo, harmony_variables, True, self.logger
                ),
                {'/latitude', '/longitude', '/rainfall_rate', '/time'},
            )

        with self.subTest('No variables, no index range subset returns none'):
            self.assertSetEqual(
                get_required_variables(self.varinfo, [], False, self.logger), set()
            )

        with self.subTest('No variables, index-range subset, returns all'):
            self.assertSetEqual(
                get_required_variables(self.varinfo, [], True, self.logger),
                all_variables,
            )
        with self.subTest('Variable configured as required in json file'):
            spl2smap_s_varinfo = VarInfoFromDmr(
                'tests/data/SC_SPL2SMAP_S.dmr', 'SPL2SMAP_S', 'hoss/hoss_config.json'
            )
            harmony_variables = [
                HarmonyVariable(
                    {
                        'fullPath': '/Soil_Moisture_Retrieval_Data_1km/albedo_1km',
                        'id': 'V1234-PROVIDER',
                        'name': 'albedo_1km',
                    }
                ),
                HarmonyVariable(
                    {
                        'fullPath': '/Soil_Moisture_Retrieval_Data_3km/soil_moisture_3km',
                        'id': 'V1234-PROVIDER',
                        'name': 'soil_moisture_3km',
                    }
                ),
            ]
            self.assertSetEqual(
                get_required_variables(
                    spl2smap_s_varinfo, harmony_variables, False, self.logger
                ),
                {
                    '/Soil_Moisture_Retrieval_Data_1km/albedo_1km',
                    '/Soil_Moisture_Retrieval_Data_3km/soil_moisture_3km',
                    '/Soil_Moisture_Retrieval_Data_1km/EASE_row_index_1km',
                    '/Soil_Moisture_Retrieval_Data_1km/EASE_column_index_1km',
                    '/Soil_Moisture_Retrieval_Data_3km/EASE_row_index_3km',
                    '/Soil_Moisture_Retrieval_Data_3km/EASE_column_index_3km',
                    '/Soil_Moisture_Retrieval_Data_1km/latitude_1km',
                    '/Soil_Moisture_Retrieval_Data_1km/longitude_1km',
                    '/Soil_Moisture_Retrieval_Data_3km/latitude_3km',
                    '/Soil_Moisture_Retrieval_Data_3km/longitude_3km',
                },
            )

    def test_fill_variables(self):
        """Ensure only the expected variables are filled (e.g., those with
        a longitude crossing the grid edge). Longitude variables should not
        themselves be filled.

        """
        varinfo = VarInfoFromDmr(
            'tests/data/rssmif16d_example.dmr',
            config_file='tests/data/test_subsetter_config.json',
        )
        input_file = 'tests/data/f16_ssmis_20200102v7.nc'
        test_file = shutil.copy(input_file, self.output_dir)
        index_ranges = {'/latitude': [0, 719], '/longitude': [1400, 10]}
        required_variables = {
            '/sst_dtime',
            '/wind_speed',
            '/latitude',
            '/longitude',
            '/time',
        }

        fill_variables(test_file, varinfo, required_variables, index_ranges)

        with Dataset(test_file, 'r') as test_output, Dataset(
            input_file, 'r'
        ) as test_input:
            # Assert none of the dimension variables are filled at any pixel
            for variable_dimension in ['/time', '/latitude', '/longitude']:
                data = test_output[variable_dimension][:]
                self.assertFalse(np.any(data.mask))
                np.testing.assert_array_equal(
                    test_input[variable_dimension], test_output[variable_dimension]
                )

            # Assert the expected range of wind_speed and sst_dtime are filled
            # but that rest of the variable matches the input file.
            for variable in ['/sst_dtime', '/wind_speed']:
                input_data = test_input[variable][:]
                output_data = test_output[variable][:]
                self.assertTrue(np.all(output_data[:][:][11:1400].mask))
                np.testing.assert_array_equal(
                    output_data[:][:][:11], input_data[:][:][:11]
                )
                np.testing.assert_array_equal(
                    output_data[:][:][1400:], input_data[:][:][1400:]
                )

            # Assert a variable that wasn't to be filled isn't
            rainfall_rate_in = test_input['/rainfall_rate'][:]
            rainfall_rate_out = test_output['/rainfall_rate'][:]
            np.testing.assert_array_equal(rainfall_rate_in, rainfall_rate_out)

    @patch('hoss.subset.Dataset')
    def test_fill_variables_no_fill(self, mock_dataset):
        """Ensure that the output file is not opened if there is no need to
        fill any variables. This will arise if:

        * There are no index ranges (e.g., a purely variable subset).
        * None of the variables cross a grid-discontinuity.

        """
        non_fill_index_ranges = {'/latitude': (100, 200), '/longitude': (150, 300)}

        test_args = [
            ['Variable subset only', {}],
            ['No index ranges need filling', non_fill_index_ranges],
        ]

        for description, index_ranges in test_args:
            with self.subTest(description):
                fill_variables(
                    self.output_dir, self.varinfo, self.required_variables, index_ranges
                )
                mock_dataset.assert_not_called()

    @patch('hoss.subset.get_fill_slice')
    def test_fill_variable(self, mock_get_fill_slice):
        """Ensure that values are only filled when the correct criteria are
        met:

        * Variable is not a longitude.
        * Variable has at least one dimension that requires filling.

        """
        fill_ranges = {'/longitude': (1439, 0)}
        dimensions_to_fill = {'/longitude'}

        # The mock return value will lead to filling the full dimension range.
        mock_get_fill_slice.return_value = slice(None)

        with self.subTest('Longitude variable should not be filled'):
            dataset_path = shutil.copy(
                'tests/data/f16_ssmis_20200102v7.nc', self.output_dir
            )

            with Dataset(dataset_path, 'a') as dataset:
                fill_variable(
                    dataset, fill_ranges, self.varinfo, '/longitude', dimensions_to_fill
                )

                self.assertFalse(dataset['/longitude'][:].any() is np.ma.masked)
                mock_get_fill_slice.assert_not_called()
                mock_get_fill_slice.reset_mock()

        with self.subTest('Variable has no dimensions needing filling'):
            dataset_path = shutil.copy(
                'tests/data/f16_ssmis_20200102v7.nc', self.output_dir
            )

            with Dataset(dataset_path, 'a') as dataset:
                fill_variable(
                    dataset, fill_ranges, self.varinfo, '/latitude', dimensions_to_fill
                )

                self.assertFalse(dataset['/latitude'][:].any() is np.ma.masked)
                mock_get_fill_slice.assert_not_called()
                mock_get_fill_slice.reset_mock()

        with self.subTest('Variable that should be filled'):
            dataset_path = shutil.copy(
                'tests/data/f16_ssmis_20200102v7.nc', self.output_dir
            )

            with Dataset(dataset_path, 'a') as dataset:
                fill_variable(
                    dataset, fill_ranges, self.varinfo, '/sst_dtime', dimensions_to_fill
                )

                self.assertTrue(dataset['/sst_dtime'][:].all() is np.ma.masked)
                mock_get_fill_slice.assert_has_calls(
                    [
                        call('/time', fill_ranges),
                        call('/latitude', fill_ranges),
                        call('/longitude', fill_ranges),
                    ]
                )
                mock_get_fill_slice.reset_mock()

    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_granule_with_no_dimensions(
        self,
        mock_get_varinfo,
        mock_get_prefetch_variables,
        mock_get_spatial_index_ranges,
        mock_get_opendap_nc4,
    ):
        """Ensure a request to extract both a variable and spatial subset for
        a granule without dimensions but with valid coordinate attributes
        without error. Because a bounding box is specified in this request,
        the prefetch functionality and the HOSS spatial_index
        functionality in `hoss.spatial.py` should be called.

        """
        harmony_message = Message(
            {'accessToken': self.access_token, 'subset': {'bbox': [2, 54, 42, 72]}}
        )
        harmony_source = Source(
            {
                'collection': 'C1268452378-EEDTEST',
                'shortName': 'SPL3SMP',
                'variables': [
                    {
                        'id': 'V1255903615-EEDTEST',
                        'name': 'surface_flag',
                        'fullPath': '/Soil_Moisture_Retrieval_Data_AM/surface_flag',
                    },
                    {
                        'id': 'V1238395077-EEDTEST',
                        'name': 'surface_flag_pm',
                        'fullPath': '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm',
                    },
                ],
            }
        )
        granule_url = 'https://harmony.earthdata.nasa.gov/bucket/spl3smp'
        collection_short_name = 'SPL3SMP'
        smap_varinfo = VarInfoFromDmr(
            'tests/data/SC_SPL3SMP_008.dmr',
            'SPL3SMP',
            'hoss/hoss_config.json',
        )
        prefetch_path = 'tests/data/SC_SPL3SMP_009_prefetch.nc4'
        subset_output_path = 'SC_SPL3SMP.009_296012210.nc4'
        required_variables = {
            '/Soil_Moisture_Retrieval_Data_AM/surface_flag',
            '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm',
            '/Soil_Moisture_Retrieval_Data_AM/latitude',
            '/Soil_Moisture_Retrieval_Data_AM/longitude',
            '/Soil_Moisture_Retrieval_Data_PM/latitude_pm',
            '/Soil_Moisture_Retrieval_Data_PM/longitude_pm',
        }

        variables_with_ranges = {
            '/Soil_Moisture_Retrieval_Data_AM/longitude[9:38][487:595]',
            '/Soil_Moisture_Retrieval_Data_PM/longitude_pm[9:38][487:595]',
            '/Soil_Moisture_Retrieval_Data_AM/latitude[9:38][487:595]',
            '/Soil_Moisture_Retrieval_Data_AM/surface_flag[9:38][487:595]',
            '/Soil_Moisture_Retrieval_Data_PM/surface_flag_pm[9:38][487:595]',
            '/Soil_Moisture_Retrieval_Data_PM/latitude_pm[9:38][487:595]',
        }
        expected_index_ranges = {
            '/Soil_Moisture_Retrieval_Data_AM/latitude_/Soil_Moisture_Retrieval_Data_AM/longitude/x_dim': (
                487,
                595,
            ),
            '/Soil_Moisture_Retrieval_Data_AM/latitude_/Soil_Moisture_Retrieval_Data_AM/longitude/y_dim': (
                9,
                38,
            ),
            '/Soil_Moisture_Retrieval_Data_PM/latitude_pm_/Soil_Moisture_Retrieval_Data_PM/longitude_pm/x_dim': (
                487,
                595,
            ),
            '/Soil_Moisture_Retrieval_Data_PM/latitude_pm_/Soil_Moisture_Retrieval_Data_PM/longitude_pm/y_dim': (
                9,
                38,
            ),
        }

        mock_get_varinfo.return_value = smap_varinfo
        mock_get_prefetch_variables.return_value = prefetch_path
        mock_get_spatial_index_ranges.return_value = expected_index_ranges
        mock_get_opendap_nc4.return_value = subset_output_path

        output_path = subset_granule(
            granule_url,
            harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertEqual(output_path, subset_output_path)
        mock_get_varinfo.assert_called_once_with(
            granule_url,
            self.output_dir,
            self.logger,
            collection_short_name,
            self.access_token,
            self.config,
        )

        mock_get_prefetch_variables.assert_called_once_with(
            granule_url,
            smap_varinfo,
            required_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )
        mock_get_spatial_index_ranges.assert_called_once_with(
            required_variables, smap_varinfo, prefetch_path, harmony_message, None
        )

        mock_get_opendap_nc4.assert_called_once_with(
            granule_url,
            variables_with_ranges,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )

    @patch('hoss.subset.get_opendap_nc4')
    @patch('hoss.subset.get_spatial_index_ranges')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_subset_granule_with_configured_dimensions(
        self,
        mock_get_varinfo,
        mock_get_prefetch_variables,
        mock_get_spatial_index_ranges,
        mock_get_opendap_nc4,
    ):
        """Ensure a request to extract both a variable and spatial subset for
        a granule with json configured dimensions but with valid coordinate attributes
        without error. Because a bounding box is specified in this request,
        the prefetch functionality and the HOSS spatial_index
        functionality in `hoss.spatial.py` should be called. The index_ranges
        should match what is expected for 2d and 3d variables.

        """
        harmony_message = Message(
            {'accessToken': self.access_token, 'subset': {'bbox': [2, 54, 42, 72]}}
        )
        harmony_source = Source(
            {
                'collection': 'C1268617120-EEDTEST',
                'shortName': 'SPL3FTP',
                'variables': [
                    {
                        'id': 'V1247777461-EEDTEST',
                        'name': 'surface_flag',
                        'fullPath': '/Freeze_Thaw_Retrieval_Data_Global/surface_flag',
                    },
                    {
                        'id': 'V1247777445-EEDTEST',
                        'name': 'transition_direction',
                        'fullPath': '/Freeze_Thaw_Retrieval_Data_Global/transition_direction',
                    },
                ],
            }
        )
        granule_url = 'https://harmony.earthdata.nasa.gov/bucket/spl3ftp'
        collection_short_name = 'SPL3FTP'
        smap_varinfo = VarInfoFromDmr(
            'tests/data/SC_SPL3FTP_004.dmr',
            'SPL3FTP',
            'hoss/hoss_config.json',
        )
        prefetch_path = 'tests/data/SC_SPL3FTP_004_prefetch.nc4'
        subset_output_path = 'SC_SPL3FTP_004_output.nc4'
        required_variables = {
            '/Freeze_Thaw_Retrieval_Data_Global/surface_flag',
            '/Freeze_Thaw_Retrieval_Data_Global/transition_direction',
            '/Freeze_Thaw_Retrieval_Data_Global/latitude',
            '/Freeze_Thaw_Retrieval_Data_Global/longitude',
        }

        variables_with_ranges = {
            '/Freeze_Thaw_Retrieval_Data_Global/longitude[][9:38][487:595]',
            '/Freeze_Thaw_Retrieval_Data_Global/latitude[][9:38][487:595]',
            '/Freeze_Thaw_Retrieval_Data_Global/surface_flag[][9:38][487:595]',
            '/Freeze_Thaw_Retrieval_Data_Global/transition_direction[9:38][487:595]',
        }
        expected_index_ranges = {
            '/Freeze_Thaw_Retrieval_Data_Global/x_dim': (487, 595),
            '/Freeze_Thaw_Retrieval_Data_Global/y_dim': (9, 38),
        }

        mock_get_varinfo.return_value = smap_varinfo
        mock_get_prefetch_variables.return_value = prefetch_path
        mock_get_spatial_index_ranges.return_value = expected_index_ranges
        mock_get_opendap_nc4.return_value = subset_output_path

        output_path = subset_granule(
            granule_url,
            harmony_source,
            self.output_dir,
            harmony_message,
            self.logger,
            self.config,
        )

        self.assertEqual(output_path, subset_output_path)
        mock_get_varinfo.assert_called_once_with(
            granule_url,
            self.output_dir,
            self.logger,
            collection_short_name,
            self.access_token,
            self.config,
        )

        mock_get_prefetch_variables.assert_called_once_with(
            granule_url,
            smap_varinfo,
            required_variables,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )
        mock_get_spatial_index_ranges.assert_called_once_with(
            required_variables, smap_varinfo, prefetch_path, harmony_message, None
        )

        mock_get_opendap_nc4.assert_called_once_with(
            granule_url,
            variables_with_ranges,
            self.output_dir,
            self.logger,
            self.access_token,
            self.config,
        )
