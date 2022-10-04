from logging import getLogger
from unittest import TestCase
from unittest.mock import Mock, patch

from harmony.exceptions import ForbiddenException, ServerException
from harmony.util import config

from pymods.exceptions import UrlAccessFailed, UrlAccessFailedWithRetries
from pymods.utilities import (download_url, format_dictionary_string,
                              format_variable_set_string,
                              get_constraint_expression, get_file_mimetype,
                              get_opendap_nc4, get_value_or_default,
                              HTTP_REQUEST_ATTEMPTS, move_downloaded_nc4)


class TestUtilities(TestCase):
    """ A class for testing functions in the pymods.utilities module. """

    @classmethod
    def setUpClass(cls):
        cls.harmony_500_error = ServerException('I can\'t do that')
        cls.harmony_auth_error = ForbiddenException('You can\'t do that.')
        cls.config = config(validate=False)
        cls.logger = getLogger('tests')

    def test_get_file_mimetype(self):
        """ Ensure a mimetype can be retrieved for a valid file path or, if
            the mimetype cannot be inferred, that the default output is
            returned. This assumes the output is a NetCDF-4 file.

        """
        with self.subTest('File with MIME type'):
            mimetype = get_file_mimetype('f16_ssmis_20200102v7.nc')
            self.assertEqual(mimetype, ('application/x-netcdf', None))

        with self.subTest('Default MIME type is returned'):
            with patch('mimetypes.guess_type') as mock_guess_type:
                mock_guess_type.return_value = (None, None)
                mimetype = get_file_mimetype('f16_ssmis_20200102v7.nc')
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
        test_data = {'dap4.ce': '%2Flatitude%3B%2Flongitude'}
        access_token = 'xyzzy'

        http_response = f'{output_directory}/output.nc'

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
                cfg=self.config
            )
            mock_util_download.reset_mock()

        with self.subTest('A request with data passes the data to Harmony.'):
            mock_util_download.return_value = http_response
            response = download_url(test_url, output_directory, self.logger,
                                    access_token, self.config, data=test_data)

            self.assertEqual(response, http_response)
            mock_util_download.assert_called_once_with(
                test_url,
                output_directory,
                self.logger,
                access_token=access_token,
                data=test_data,
                cfg=self.config
            )
            mock_util_download.reset_mock()

        with self.subTest('500 error triggers a retry.'):
            mock_util_download.side_effect = [self.harmony_500_error,
                                              http_response]

            response = download_url(test_url, output_directory, self.logger)

            self.assertEqual(response, http_response)
            self.assertEqual(mock_util_download.call_count, 2)
            mock_util_download.reset_mock()

        with self.subTest('Non-500 error does not retry, and is re-raised.'):
            mock_util_download.side_effect = [self.harmony_auth_error,
                                              http_response]

            with self.assertRaises(UrlAccessFailed):
                download_url(test_url, output_directory, self.logger,
                             access_token, self.config)

            mock_util_download.assert_called_once_with(
                test_url,
                output_directory,
                self.logger,
                access_token=access_token,
                data=None,
                cfg=self.config
            )
            mock_util_download.reset_mock()

        with self.subTest('Maximum number of attempts not exceeded.'):
            mock_util_download.side_effect = [
                self.harmony_500_error
            ] * (HTTP_REQUEST_ATTEMPTS + 1)
            with self.assertRaises(UrlAccessFailedWithRetries):
                download_url(test_url, output_directory, self.logger)

            self.assertEqual(mock_util_download.call_count,
                             HTTP_REQUEST_ATTEMPTS)
            mock_util_download.reset_mock()

    @patch('pymods.utilities.move_downloaded_nc4')
    @patch('pymods.utilities.util_download')
    def test_get_opendap_nc4(self, mock_download, mock_move_download):
        """ Ensure a request is sent to OPeNDAP that combines the URL of the
            granule with a constraint expression.

            Once the request is completed, the output file should be moved to
            ensure a second request to the same URL is still performed.

        """
        downloaded_file_name = 'output_file.nc4'
        moved_file_name = 'moved_file.nc4'
        mock_download.return_value = downloaded_file_name
        mock_move_download.return_value = moved_file_name

        url = 'https://opendap.earthdata.nasa.gov/granule'
        required_variables = {'variable'}
        output_dir = '/path/to/temporary/folder/'
        access_token = 'secret_token!!!'
        expected_data = {'dap4.ce': 'variable'}

        with self.subTest('Request with variables includes dap4.ce'):
            output_file = get_opendap_nc4(url, required_variables, output_dir,
                                          self.logger, access_token,
                                          self.config)

            self.assertEqual(output_file, moved_file_name)
            mock_download.assert_called_once_with(
                f'{url}.dap.nc4', output_dir, self.logger,
                access_token=access_token, data=expected_data, cfg=self.config
            )
            mock_move_download.assert_called_once_with(output_dir,
                                                       downloaded_file_name)

        mock_download.reset_mock()
        mock_move_download.reset_mock()

        with self.subTest('Request with no variables omits dap4.ce'):
            output_file = get_opendap_nc4(url, {}, output_dir, self.logger,
                                          access_token, self.config)

            self.assertEqual(output_file, moved_file_name)
            mock_download.assert_called_once_with(
                f'{url}.dap.nc4', output_dir, self.logger,
                access_token=access_token, data=None, cfg=self.config
            )
            mock_move_download.assert_called_once_with(output_dir,
                                                       downloaded_file_name)

    def test_get_constraint_expression(self):
        """ Ensure a correctly encoded DAP4 constraint expression is
            constructed for the given input.

            URL encoding:

            - %2F = '/'
            - %3A = ':'
            - %3B = ';'
            - %5B = '['
            - %5D = ']'

            Note - with sets, the order can't be guaranteed, so there are two
            options for the combined constraint expression.

        """
        with self.subTest('No index ranges specified'):
            self.assertIn(
                get_constraint_expression({'/alpha_var', '/blue_var'}),
                ['%2Falpha_var%3B%2Fblue_var', '%2Fblue_var%3B%2Falpha_var']
            )

        with self.subTest('Variables with index ranges'):
            self.assertIn(
                get_constraint_expression({'/alpha_var[1:2]', '/blue_var[3:4]'}),
                ['%2Falpha_var%5B1%3A2%5D%3B%2Fblue_var%5B3%3A4%5D',
                 '%2Fblue_var%5B3%3A4%5D%3B%2Falpha_var%5B1%3A2%5D']
            )

    @patch('pymods.utilities.move')
    @patch('pymods.utilities.uuid4')
    def test_move_downloaded_nc4(self, mock_uuid4, mock_move):
        """ Ensure a specified file is moved to the specified location. """
        mock_uuid4.return_value = Mock(hex='uuid4')
        output_dir = '/tmp/path/to'
        old_path = '/tmp/path/to/file.nc4'

        self.assertEqual(move_downloaded_nc4(output_dir, old_path),
                         '/tmp/path/to/uuid4.nc4')

        mock_move.assert_called_once_with('/tmp/path/to/file.nc4',
                                          '/tmp/path/to/uuid4.nc4')

    def test_format_variable_set(self):
        """ Ensure a set of variable strings is printed out as expected, and
            does not contain any curly braces.

            The test is a little convoluted, because sets are unordered, so the
            exact ordering of the variables within the string may not be
            identical.

        """
        variable_set = {'/var_one', '/var_two', '/var_three'}
        formatted_string = format_variable_set_string(variable_set)

        self.assertNotIn('{', formatted_string)
        self.assertNotIn('}', formatted_string)
        self.assertSetEqual(variable_set, set(formatted_string.split(', ')))

    def test_format_dictionary_string(self):
        """ Ensure a dictionary is formatted to a string without curly braces.
            This function assumes only a single level dictionary, without any
            sets for values.

        """
        input_dictionary = {'key_one': 'value_one', 'key_two': 'value_two'}

        self.assertEqual(format_dictionary_string(input_dictionary),
                         'key_one: value_one\nkey_two: value_two')

    def test_get_value_or_default(self):
        """ Ensure a value is retrieved if supplied, even if it is 0, or a
            default value is returned if not.

        """
        with self.subTest('Value is returned'):
            self.assertEqual(get_value_or_default(10, 20), 10)

        with self.subTest('Value = 0 is returned'):
            self.assertEqual(get_value_or_default(0, 20), 0)

        with self.subTest('Value = None returns the supplied default'):
            self.assertEqual(get_value_or_default(None, 20), 20)
