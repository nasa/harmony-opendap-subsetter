from logging import Logger
from unittest import TestCase
from unittest.mock import patch
from urllib.error import HTTPError

from harmony.util import config

from pymods.exceptions import UrlAccessFailed, UrlAccessFailedWithRetries
from pymods.utilities import (download_url, get_file_mimetype,
                              HTTP_REQUEST_ATTEMPTS)


class TestUtilities(TestCase):
    """ A class for testing functions in the pymods.utilities module. """

    @classmethod
    def setUpClass(cls):
        cls.namespace = 'namespace_string'

    def setUp(self):
        self.logger = Logger('tests')
        self.config = config(validate=False)

    def test_get_file_mimetype(self):
        """ Ensure a mimetype can be retrieved for a valid file path or, if
            the mimetype cannot be inferred, that the default output is
            returned. This assumes the output is a NetCDF-4 file.

        """
        with self.subTest('File with MIME type'):
            mimetype = get_file_mimetype('africa.nc')
            self.assertEqual(mimetype, ('application/x-netcdf', None))

        with self.subTest('Default MIME type is returned'):
            with patch('mimetypes.guess_type') as mock_guess_type:
                mock_guess_type.return_value = (None, None)
                mimetype = get_file_mimetype('africa.nc')
                self.assertEqual(mimetype, ('application/x-netcdf4', None))

    @patch('pymods.utilities.util_download')
    def test_download_url(self, mock_util_download):
        """ Ensure that the `harmony.util.download` function is called. Also
            ensure that if a 500 error is returned, the request is retried. If
            a different HTTPError occurs, the caught HTTPError should be
            re-raised. Finally, check the maximum number of request attempts is
            not exceeded.

        """
        output_directory = 'output/dir'
        test_url = 'test.org'
        access_token = 'xyzzy'
        message_retry = 'Internal Server Error'
        message_other = 'Authentication Error'

        http_response = f'{output_directory}/output.nc'
        http_error_retry = HTTPError(test_url, 500, message_retry, {}, None)
        http_error_other = HTTPError(test_url, 403, message_other, {}, None)

        with self.subTest('Successful response, only make one request.'):
            mock_util_download.return_value = http_response
            response = download_url(test_url, output_directory, self.logger,
                                    access_token, self.config)

            self.assertEqual(response, http_response)
            mock_util_download.assert_called_once_with(
                test_url,
                output_directory,
                self.logger,
                access_token=access_token,
                data=None,
                cfg=self.config)

            mock_util_download.reset_mock()

        with self.subTest('500 error triggers a retry.'):
            mock_util_download.side_effect = [http_error_retry, http_response]

            response = download_url(test_url, output_directory, self.logger)

            self.assertEqual(response, http_response)
            self.assertEqual(mock_util_download.call_count, 2)

        with self.subTest('Non-500 error does not retry, and is re-raised.'):
            mock_util_download.side_effect = [http_error_other, http_response]

            with self.assertRaises(UrlAccessFailed):
                download_url(test_url, output_directory, self.logger)
                mock_util_download.assert_called_once_with(test_url,
                                                           output_directory,
                                                           self.logger)

        with self.subTest('Maximum number of attempts not exceeded.'):
            mock_util_download.side_effect = [http_error_retry] * (HTTP_REQUEST_ATTEMPTS + 1)
            with self.assertRaises(UrlAccessFailedWithRetries):
                download_url(test_url, output_directory, self.logger)
                self.assertEqual(mock_util_download.call_count,
                                 HTTP_REQUEST_ATTEMPTS)
