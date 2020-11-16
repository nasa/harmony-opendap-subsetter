from logging import Logger
from unittest import TestCase
from unittest.mock import patch
import json

from harmony.message import Message
import harmony.util

from pymods.subset import subset_granule
from tests.utilities import contains


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
        self.logger = Logger('tests')
        self.config = harmony.util.config(validate=False)

    @patch('pymods.subset.VarInfo')
    @patch('pymods.subset.download_url')
    def test_subset_granule(self, mock_download_url, mock_var_info_dmr):
        """ Ensure valid request does not raise exception,
            raise appropriate exception otherwise.
            Note: %2F is a URL encoded slash and %3B is a URL encoded semi-colon.

        """
        granule = self.message.granules[0]
        mock_download_url.return_value = 'africa_subset.nc4'

        # Note: return value below is a list, not a set, so the order can be
        # guaranteed in the assertions that the request to OPeNDAP was made
        # with all required variables.
        mock_var_info_dmr.return_value.get_required_variables.return_value = [
            '/alpha_var', '/blue_var'
        ]

        with self.subTest('Succesful calls to OPeNDAP'):
            output_path = subset_granule(granule, self.logger)
            mock_download_url.assert_called_once_with(
                f'{self.granule_url}.dap.nc4?dap4.ce=%2Falpha_var%3B%2Fblue_var',
                contains('/tmp/tmp'),
                self.logger,
                data='',
                access_token=None,
                config=None
            )
            self.assertIn('africa_subset.nc4', output_path)
