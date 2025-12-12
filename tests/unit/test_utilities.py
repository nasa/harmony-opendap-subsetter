from logging import getLogger
from unittest import TestCase
from unittest.mock import Mock, patch

from harmony_service_lib.exceptions import ForbiddenException, ServerException
from harmony_service_lib.util import config

from hoss.exceptions import UrlAccessFailed
from hoss.harmony_log_context import set_logger
from hoss.utilities import (
    download_url,
    format_dictionary_string,
    format_variable_set_string,
    get_constraint_expression,
    get_file_mimetype,
    get_opendap_nc4,
    get_value_or_default,
    move_downloaded_nc4,
    unexecuted_url_requested,
)


class TestUtilities(TestCase):
    """A class for testing functions in the hoss.utilities module."""

    @classmethod
    def setUpClass(cls):
        cls.harmony_500_error = ServerException('I can\'t do that')
        cls.harmony_auth_error = ForbiddenException('You can\'t do that.')
        cls.config = config(validate=False)
        cls.logger = getLogger('test')
        set_logger(cls.logger)

    def test_get_file_mimetype(self):
        """Ensure a mimetype can be retrieved for a valid file path or, if
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

    @patch('hoss.utilities.util_download')
    def test_download_url(self, mock_util_download):
        """Ensure that the `harmony.util.download` function is called. If an
        error occurs, the caught exception should be re-raised with a
        custom exception with a human-readable error message.

        """
        output_directory = 'output/dir'
        test_url = 'fake_website.com'
        test_data = {'dap4.ce': '%2Flatitude%3B%2Flongitude'}
        access_token = 'xyzzy'

        http_response = f'{output_directory}/output.nc'

        with self.subTest('Successful response, only make one request.'):
            mock_util_download.return_value = http_response
            response = download_url(
                test_url, output_directory, access_token, self.config
            )

            self.assertEqual(response, http_response)
            mock_util_download.assert_called_once_with(
                test_url,
                output_directory,
                self.logger,
                access_token=access_token,
                data=None,
                cfg=self.config,
            )
            mock_util_download.reset_mock()

        with self.subTest('A request with data passes the data to Harmony.'):
            mock_util_download.return_value = http_response
            response = download_url(
                test_url,
                output_directory,
                access_token,
                self.config,
                data=test_data,
            )

            self.assertEqual(response, http_response)
            mock_util_download.assert_called_once_with(
                test_url,
                output_directory,
                self.logger,
                access_token=access_token,
                data=test_data,
                cfg=self.config,
            )
            mock_util_download.reset_mock()

        with self.subTest('500 error is caught and handled.'):
            mock_util_download.side_effect = [self.harmony_500_error, http_response]

            with self.assertRaises(UrlAccessFailed):
                download_url(test_url, output_directory, access_token, self.config)

            mock_util_download.assert_called_once_with(
                test_url,
                output_directory,
                self.logger,
                access_token=access_token,
                data=None,
                cfg=self.config,
            )
            mock_util_download.reset_mock()

        with self.subTest('Non-500 error does not retry, and is re-raised.'):
            mock_util_download.side_effect = [self.harmony_auth_error, http_response]

            with self.assertRaises(UrlAccessFailed):
                download_url(test_url, output_directory, access_token, self.config)

            mock_util_download.assert_called_once_with(
                test_url,
                output_directory,
                self.logger,
                access_token=access_token,
                data=None,
                cfg=self.config,
            )
            mock_util_download.reset_mock()

    @patch('hoss.utilities.move_downloaded_nc4')
    @patch('hoss.utilities.util_download')
    def test_get_opendap_nc4(self, mock_download, mock_move_download):
        """Ensure a request is sent to OPeNDAP that combines the URL of the
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

        with self.subTest('Request with OPeNDAP URL mimetype'):
            opendap_url_mimetype = 'application/x-netcdf4;profile="opendap_url"'
            output_url = get_opendap_nc4(
                url,
                required_variables,
                output_dir,
                access_token,
                self.config,
                opendap_url_mimetype,
            )
            self.assertEqual(
                'https://opendap.earthdata.nasa.gov/granule.dap.nc4?dap4.ce=variable',
                output_url,
            )
            mock_download.assert_not_called()
            mock_move_download.assert_not_called()

        mock_download.reset_mock()
        mock_move_download.reset_mock()

        with self.subTest('Request with variables includes dap4.ce'):
            output_file = get_opendap_nc4(
                url,
                required_variables,
                output_dir,
                access_token,
                self.config,
                'fake-mimetype',
            )

            self.assertEqual(output_file, moved_file_name)
            mock_download.assert_called_once_with(
                f'{url}.dap.nc4',
                output_dir,
                self.logger,
                access_token=access_token,
                data=expected_data,
                cfg=self.config,
            )
            mock_move_download.assert_called_once_with(output_dir, downloaded_file_name)

        mock_download.reset_mock()
        mock_move_download.reset_mock()

        with self.subTest('Request with no variables omits dap4.ce'):
            output_file = get_opendap_nc4(
                url, {}, output_dir, access_token, self.config
            )

            self.assertEqual(output_file, moved_file_name)
            mock_download.assert_called_once_with(
                f'{url}.dap.nc4',
                output_dir,
                self.logger,
                access_token=access_token,
                data=None,
                cfg=self.config,
            )
            mock_move_download.assert_called_once_with(output_dir, downloaded_file_name)

    def test_get_constraint_expression(self):
        """Ensure a correctly encoded DAP4 constraint expression is
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
                ['%2Falpha_var%3B%2Fblue_var', '%2Fblue_var%3B%2Falpha_var'],
            )

        with self.subTest('Variables with index ranges'):
            self.assertIn(
                get_constraint_expression({'/alpha_var[1:2]', '/blue_var[3:4]'}),
                [
                    '%2Falpha_var%5B1%3A2%5D%3B%2Fblue_var%5B3%3A4%5D',
                    '%2Fblue_var%5B3%3A4%5D%3B%2Falpha_var%5B1%3A2%5D',
                ],
            )

    @patch('hoss.utilities.move')
    @patch('hoss.utilities.uuid4')
    def test_move_downloaded_nc4(self, mock_uuid4, mock_move):
        """Ensure a specified file is moved to the specified location."""
        mock_uuid4.return_value = Mock(hex='uuid4')
        output_dir = '/tmp/path/to'
        old_path = '/tmp/path/to/file.nc4'

        self.assertEqual(
            move_downloaded_nc4(output_dir, old_path), '/tmp/path/to/uuid4.nc4'
        )

        mock_move.assert_called_once_with(
            '/tmp/path/to/file.nc4', '/tmp/path/to/uuid4.nc4'
        )

    def test_format_variable_set(self):
        """Ensure a set of variable strings is printed out as expected, and
        does not contain any curly braces.

        The formatted string is broken up for verification because sets are
        unordered, so the exact ordering of the variables within the
        formatted string may not be consistent between runs.

        """
        variable_set = {'/var_one', '/var_two', '/var_three'}
        formatted_string = format_variable_set_string(variable_set)

        self.assertNotIn('{', formatted_string)
        self.assertNotIn('}', formatted_string)
        self.assertSetEqual(variable_set, set(formatted_string.split(', ')))

    def test_format_dictionary_string(self):
        """Ensure a dictionary is formatted to a string without curly braces.
        This function assumes only a single level dictionary, without any
        sets for values.

        """
        input_dictionary = {'key_one': 'value_one', 'key_two': 'value_two'}

        self.assertEqual(
            format_dictionary_string(input_dictionary),
            'key_one: value_one\nkey_two: value_two',
        )

    def test_get_value_or_default(self):
        """Ensure a value is retrieved if supplied, even if it is 0, or a
        default value is returned if not.

        """
        with self.subTest('Value is returned'):
            self.assertEqual(get_value_or_default(10, 20), 10)

        with self.subTest('Value = 0 is returned'):
            self.assertEqual(get_value_or_default(0, 20), 0)

        with self.subTest('Value = None returns the supplied default'):
            self.assertEqual(get_value_or_default(None, 20), 20)

    def test_unexecuted_url_requested(self):
        """Ensure that True is returned when a valid OPeNDAP URL format
        string is in the Harmony message, otherwise False should be returned.

        """
        with self.subTest('Valid opendap_url format 1'):
            self.assertTrue(
                unexecuted_url_requested('application/x-netcdf4; profile="opendap_url"')
            )

        with self.subTest('Valid opendap_url format 2'):
            self.assertTrue(
                unexecuted_url_requested('application/x-netcdf4;profile="opendap_url"')
            )

        with self.subTest('Valid opendap_url format 3'):
            self.assertTrue(
                unexecuted_url_requested('application/x-netcdf4;profile=opendap_url')
            )

        with self.subTest('NetCDF4 format'):
            self.assertFalse(unexecuted_url_requested('application/x-netcdf4'))

        with self.subTest('Some other format'):
            self.assertFalse(unexecuted_url_requested('fake-mimetype'))
