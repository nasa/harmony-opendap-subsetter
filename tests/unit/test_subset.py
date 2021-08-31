from logging import Logger
from unittest import TestCase
from unittest.mock import ANY, Mock, patch
import json
import shutil
from tempfile import mkdtemp

from harmony.message import Message
from harmony.util import config

from pymods.subset import subset_granule


class TestSubset(TestCase):
    """ Test the module that performs subsetting on a single granule. """

    @classmethod
    def setUpClass(cls):
        cls.granule_url = 'https://harmony.earthdata.nasa.gov/bucket/africa'
        cls.message_content = {
            'sources': [{'collection': 'C1233860183-EEDTEST',
                         'variables': [{'id': 'V1234834148-EEDTEST',
                                        'name': 'alpha_var',
                                        'fullPath': 'alpha_var'}],
                         'granules': [{'id': 'G1233860471-EEDTEST',
                                       'url': cls.granule_url}]}]
        }
        cls.message = Message(json.dumps(cls.message_content))

    def setUp(self):
        self.output_dir = mkdtemp()
        self.logger = Logger('tests')
        self.config = config(validate=False)

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    @patch('pymods.subset.get_geo_bounding_box_subset')
    @patch('pymods.subset.VarInfoFromDmr')
    @patch('pymods.subset.get_opendap_nc4')
    @patch('pymods.subset.download_url')
    def test_subset_granule_no_geo(self, mock_download_dmr, mock_get_opendap_data,
                                   mock_var_info_dmr, mock_get_geo_subset):
        """ Ensure valid request does not raise exception,
            raise appropriate exception otherwise.
            Note: %2F is a URL encoded slash and %3B is a URL encoded semi-colon.

            Because no bounding box is specified in this request, the HOSS
            functionality in `pymods.geo_grid.py` should not be called.

        """
        url = self.granule_url
        mock_download_dmr.return_value = 'africa_subset.dmr'
        mock_get_opendap_data.return_value = 'africa_subset.nc4'

        mock_var_info_dmr.return_value.get_required_variables.return_value = {
            '/alpha_var', '/blue_var'
        }
        variables = self.message.sources[0].variables

        output_path = subset_granule(url, variables, self.output_dir,
                                     self.logger, access_token='access',
                                     config=self.config)

        mock_download_dmr.assert_called_once_with(f'{url}.dmr',
                                                  ANY,
                                                  self.logger,
                                                  access_token='access',
                                                  config=self.config)
        mock_get_opendap_data.assert_called_once_with(url,
                                                      {'/alpha_var', '/blue_var'},
                                                      self.output_dir,
                                                      self.logger,
                                                      'access',
                                                      self.config)
        self.assertIn('africa_subset.nc4', output_path)
        mock_get_geo_subset.assert_not_called()

    @patch('pymods.subset.get_geo_bounding_box_subset')
    @patch('pymods.subset.VarInfoFromDmr')
    @patch('pymods.subset.get_opendap_nc4')
    @patch('pymods.subset.download_url')
    def test_subset_granule_geo(self, mock_download_dmr, mock_get_opendap_data,
                                mock_var_info_dmr, mock_get_geo_subset):
        """ Ensure valid request does not raise exception,
            raise appropriate exception otherwise.
            Note: %2F is a URL encoded slash and %3B is a URL encoded semi-colon.

            Because a bounding box is specified in this request, the HOSS
            functionality in `pymods.geo_grid.py` should be called instead of
            the `pymods.utilities.get_opendap_nc4` function directly.

        """
        url = self.granule_url
        required_variables = {'/alpha_var', '/blue_var'}
        bounding_box = [40, -30, 50, -20]
        mock_download_dmr.return_value = 'africa_subset.dmr'
        mock_get_geo_subset.return_value = 'africa_geo_subset.nc4'

        mock_var_info = Mock()
        mock_var_info.get_required_variables.return_value = required_variables
        mock_var_info_dmr.return_value = mock_var_info

        variables = self.message.sources[0].variables

        output_path = subset_granule(url, variables, self.output_dir,
                                     self.logger, access_token='access',
                                     config=self.config,
                                     bounding_box=bounding_box)

        mock_download_dmr.assert_called_once_with(f'{url}.dmr',
                                                  self.output_dir,
                                                  self.logger,
                                                  access_token='access',
                                                  config=self.config)
        mock_get_geo_subset.assert_called_once_with({'/alpha_var', '/blue_var'},
                                                    mock_var_info,
                                                    bounding_box, url,
                                                    self.output_dir,
                                                    self.logger, 'access',
                                                    self.config)
        self.assertIn('africa_geo_subset.nc4', output_path)
        mock_get_opendap_data.assert_not_called()

    @patch('pymods.subset.get_opendap_nc4')
    @patch('pymods.subset.get_geo_bounding_box_subset')
    @patch('pymods.subset.download_url')
    def test_subset_non_geo_no_variables(self, mock_download_dmr,
                                         mock_get_geo_subset,
                                         mock_get_opendap_data):
        """ Ensure a request without a bounding box and without any specified
            variables will produce a request to OPeNDAP that does not specify
            any variables. This will default to retrieving the full NetCDF-4
            file from OPeNDAP.

        """
        url = self.granule_url
        mock_download_dmr.return_value = 'tests/data/rssmif16d_example.dmr'
        mock_get_opendap_data.return_value = 'rssmif16d_subset.nc4'

        output_path = subset_granule(url, [], self.output_dir, self.logger,
                                     access_token='access', config=self.config,
                                     bounding_box=None)

        self.assertIn('rssmif16d_subset.nc4', output_path)
        mock_download_dmr.assert_called_once_with(f'{url}.dmr',
                                                  self.output_dir,
                                                  self.logger,
                                                  access_token='access',
                                                  config=self.config)
        mock_get_geo_subset.assert_not_called()
        mock_get_opendap_data.assert_called_once_with(url, set(),
                                                      self.output_dir,
                                                      self.logger, 'access',
                                                      self.config)

    @patch('pymods.subset.get_opendap_nc4')
    @patch('pymods.subset.get_geo_bounding_box_subset')
    @patch('pymods.subset.download_url')
    def test_subset_geo_no_variables(self, mock_download_dmr,
                                     mock_get_geo_subset,
                                     mock_get_opendap_data):
        """ Ensure a request with a bounding box, but without any specified
            variables will consider all science and metadata variables as the
            requested variables. This situation will arise if a user requests
            all variables. HOSS will need to explicitly list all the variables
            it retrieves (unlike the variable subsetter) as the DAP4 constraint
            expression will need to specify index ranges for all geographically
            gridded variables.

        """
        url = self.granule_url
        bounding_box = [40, -30, 50, -20]
        expected_variables = {'/atmosphere_cloud_liquid_water_content',
                              '/atmosphere_water_vapor_content',
                              '/latitude', '/longitude', '/rainfall_rate',
                              '/sst_dtime', '/time', '/wind_speed'}
        mock_download_dmr.return_value = 'tests/data/rssmif16d_example.dmr'
        mock_get_geo_subset.return_value = 'rssmif16d_subset.nc4'

        output_path = subset_granule(url, [], self.output_dir, self.logger,
                                     access_token='access', config=self.config,
                                     bounding_box=bounding_box)

        self.assertIn('rssmif16d_subset.nc4', output_path)
        mock_download_dmr.assert_called_once_with(f'{url}.dmr',
                                                  self.output_dir,
                                                  self.logger,
                                                  access_token='access',
                                                  config=self.config)
        mock_get_geo_subset.assert_called_once_with(expected_variables,
                                                    ANY, bounding_box,
                                                    url, self.output_dir,
                                                    self.logger, 'access',
                                                    self.config)
        mock_get_opendap_data.assert_not_called()

    @patch('pymods.subset.get_opendap_nc4')
    @patch('pymods.subset.get_geo_bounding_box_subset')
    @patch('pymods.subset.download_url')
    def test_subset_non_variable_dimensions(self, mock_download_dmr,
                                            mock_get_geo_subset,
                                            mock_get_opendap_data):
        """ Ensure a request with a bounding box, without specified variables,
            will not include non-variable dimensions in the requests to OPeNDAP
            in the list of required variables.

            In the GPM_3IMERGHH data, the specific dimensions that should not
            be included in the required variables are `latv`, `lonv` and `nv`.
            These are size-only dimensions for the `lat_bnds`, `lon_bnds` and
            `time_bnds` variables.

        """
        url = self.granule_url
        bounding_box = [40, -30, 50, -20]
        expected_variables = {
            '/Grid/HQobservationTime', '/Grid/HQprecipitation',
            '/Grid/HQprecipSource', '/Grid/IRkalmanFilterWeight',
            '/Grid/IRprecipitation', '/Grid/lat', '/Grid/lat_bnds',
            '/Grid/lon', '/Grid/lon_bnds', '/Grid/precipitationCal',
            '/Grid/precipitationQualityIndex', '/Grid/precipitationUncal',
            '/Grid/probabilityLiquidPrecipitation', '/Grid/randomError',
            '/Grid/time', '/Grid/time_bnds'
        }

        mock_download_dmr.return_value = 'tests/data/GPM_3IMERGHH_example.dmr'
        mock_get_geo_subset.return_value = 'GPM_3IMERGHH_subset.nc4'

        output_path = subset_granule(url, [], self.output_dir, self.logger,
                                     access_token='access', config=self.config,
                                     bounding_box=bounding_box)

        self.assertIn('GPM_3IMERGHH_subset.nc4', output_path)
        mock_download_dmr.assert_called_once_with(f'{url}.dmr',
                                                  self.output_dir,
                                                  self.logger,
                                                  access_token='access',
                                                  config=self.config)
        mock_get_geo_subset.assert_called_once_with(expected_variables,
                                                    ANY, bounding_box,
                                                    url, self.output_dir,
                                                    self.logger, 'access',
                                                    self.config)
        mock_get_opendap_data.assert_not_called()
