from logging import Logger
from unittest import TestCase
from unittest.mock import patch, ANY
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

    @patch('pymods.subset.VarInfo')
    @patch('pymods.subset.download_url')
    def test_subset_granule(self, mock_download_url, mock_var_info_dmr):
        """ Ensure valid request does not raise exception,
            raise appropriate exception otherwise.
            Note: %2F is a URL encoded slash and %3B is a URL encoded semi-colon.

        """
        url = self.__class__.granule_url
        mock_download_url.return_value = 'africa_subset.nc4'

        # Note: return value below is a list, not a set, so the order can be
        # guaranteed in the assertions that the request to OPeNDAP was made
        # with all required variables.
        mock_var_info_dmr.return_value.get_required_variables.return_value = [
            '/alpha_var', '/blue_var'
        ]
        variables = self.message.sources[0].variables

        with self.subTest('Succesful calls to OPeNDAP'):
            output_path = subset_granule(url, variables, self.output_dir, self.logger)
            print(mock_download_url.mock_calls)
            mock_download_url.assert_called_once_with(
                f'{url}.dap.nc4?dap4.ce=%2Falpha_var%3B%2Fblue_var',
                ANY,
                self.logger,
                access_token=None,
                config=None
            )
            self.assertIn('africa_subset.nc4', output_path)
