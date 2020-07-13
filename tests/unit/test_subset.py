from logging import Logger
from unittest import TestCase
from unittest.mock import patch, MagicMock
import json

from requests.exceptions import HTTPError
from harmony.message import Message

from pymods.subset import subset_granule
from pymods.var_info import VarInfo
from tests.utilities import MockResponse


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

    @patch('pymods.subset.VarInfo')
    @patch('pymods.subset.get_url_response')
    def test_subset_granule(self, mock_get, mock_var_info):
        """ Ensure valid request does not raise exception,
            raise appropriate exception otherwise.
        """
        granule = self.message.granules[0]
        mock_get.return_value = MockResponse(200, b'CONTENT')

        # Note: return value below is a list, not a set, so the order can be
        # guaranteed in the assertions that the request to OPeNDAP was made
        # with all required variables.
        mock_var_info.return_value.get_required_variables.return_value = [
            '/alpha_var', '/blue_var'
        ]

        with self.subTest('Succesful calls to OPeNDAP'):
            output_path = subset_granule(granule, self.logger)
            mock_get.assert_called_once_with(
                f'{self.granule_url}.nc4?alpha_var,blue_var',
                self.logger
            )
            self.assertIn('africa_subset.nc', output_path)

        with self.subTest('Unauthorized error'):
            with self.assertRaises(HTTPError):
                mock_get.side_effect = HTTPError("Request cannot be completed with error code 400")
                subset_granule(granule, self.logger)
