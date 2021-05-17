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
