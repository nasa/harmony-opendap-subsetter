from logging import Logger
from unittest import TestCase
from unittest.mock import patch, Mock
import json
from requests.exceptions import HTTPError

from harmony.message import Message
from pymods.subset import subset_granule
from tests.utilities import contains

def generate_response(status=200,
                      content=b'CONTENT',
                      json_data=None,
                      raise_for_status=None):
    mock_resp = Mock()
    mock_resp.raise_for_status = Mock()
    if raise_for_status:
        mock_resp.raise_for_status.side_effect = raise_for_status
    mock_resp.status_code = status
    mock_resp.content = content
    if json_data:
        mock_resp.json = Mock(return_value=json_data)

    return mock_resp


class TestSubset(TestCase):
    """ Test the module that performs subsetting on a single granule. """

    @classmethod
    def setUpClass(cls):
        cls.message_content = ({'sources': [{'collection': 'C1233860183-EEDTEST',
                                             'variables': [{'id': 'V1234834148-EEDTEST',
                                                            'name': 'alpha_var',
                                                            'fullPath': 'alpha_var'}],
                                             'granules': [{'id': 'G1233860471-EEDTEST',
                                                           'url': '/home/tests/data/africa.nc'}]
                                             }]})

        cls.message = Message(json.dumps(cls.message_content))

    def setUp(self):
        self.logger = Logger('tests')

    @patch('pymods.subset.get_token')
    @patch('pymods.subset.cmr_query')
    @patch('pymods.subset.VarInfo')
    @patch('pymods.subset.requests.get')
    def test_subset_granule(self, mock_get, mock_var_info, mock_cmr_query, mock_get_token):
        """ Ensure valid request does not raise exception,
            raise appropriate exception otherwise.
        """
        granule = self.message.granules[0]
        granule.local_filename = '/home/tests/data/africa.nc'
        mock_cmr_query.side_effect = ['entry_title', 'granule ur']
        mock_response = generate_response()
        mock_get.return_value = mock_response
        mock_get_token.return_value = 'token'
        mock_var_info = Mock()

        output_path = subset_granule(granule, self.logger)
        mock_get_token.assert_called_once()
        mock_cmr_query.assert_called()
        mock_get.assert_called_once()
        self.assertIn('africa_subset.nc', output_path)

        with self.subTest('Unauthorized error'):
            with self.assertRaises(HTTPError):
                mock_cmr_query.side_effect = ['entry_title', 'granule ur']
                mock_response = generate_response(status=401,
                                                  raise_for_status=HTTPError(
                                                      "Request cannot be completed with error code 400"))
                mock_get.return_value = mock_response
                subset_granule(granule, self.logger)

        with self.subTest('Service Unavailable'):
            with self.assertRaises(HTTPError):
                mock_cmr_query.side_effect = ['entry_title', 'granule ur']
                mock_response = generate_response(status=500,
                                                  raise_for_status=HTTPError(
                                                      "Request cannot be completed with error code 400"))
                mock_get.return_value = mock_response
                subset_granule(granule, self.logger)
