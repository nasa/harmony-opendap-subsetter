from logging import Logger
from unittest import TestCase
from unittest.mock import patch, Mock
import os
import xml.etree.ElementTree as ET

from requests import HTTPError
import numpy as np

from pymods.exceptions import AuthorizationError, DmrNamespaceError
from pymods.utilities import (create_netrc_file, get_file_mimetype,
                              get_url_response, get_xml_attribute,
                              get_xml_namespace, pydap_attribute_path,
                              recursive_get)
from tests.utilities import MockResponse


class TestUtilities(TestCase):
    """ A class for testing functions in the pymods.utilities module. """

    @classmethod
    def setUpClass(cls):
        cls.namespace = 'namespace_string'

    def setUp(self):
        self.logger = Logger('tests')

    def test_get_file_mimetype(self):
        """ Ensure a mimetype can be retrieved for a valid file path or, if
            the mimetype cannot be inferred, that the default output is
            returned.

        """
        with self.subTest('File with MIME type'):
            mimetype = get_file_mimetype('africa.nc')
            self.assertEqual(mimetype, ('application/x-netcdf', None))

        with self.subTest('Default MIME type is returned'):
            with patch('mimetypes.guess_type') as mock_guess_type:
                mock_guess_type.return_value = (None, None)
                mimetype = get_file_mimetype('africa.nc')
                self.assertEqual(mimetype, ('application/octet-stream', None))

    def test_recursive_get(self):
        """ Can retrieve a nested dictionary value, or account for missing
            data.

        """
        test_args = [
            ['Top level', {'a': 'b'}, ['a'], 'b'],
            ['Nested', {'a': {'b': 'c'}}, ['a', 'b'], 'c'],
            ['Missing nested data', {'a': {'c': 'd'}}, ['a', 'b'], None],
            ['Missing top level', {'b': {'c': 'd'}}, ['a', 'c'], None]
        ]

        for description, test_dictionary, keys, expected_output in test_args:
            with self.subTest(description):
                self.assertEqual(recursive_get(test_dictionary, keys),
                                 expected_output)

    def test_pydap_attribute_path(self):
        """ Check that a fully qualified path to a metadata attribute is
            correctly converted to a combination of two keys, to locate the
            attribute in the pydap global attributes for the granule.

            For example: /Metadata/SeriesIdentification/shortName will be
            located at: dataset.attributes['Metadata_SeriesIdentification']['shortName']

        """
        test_args = [['Not nested', '/short_name', ['short_name']],
                     ['Singly nested', '/Metadata/short_name',
                      ['Metadata', 'short_name']],
                     ['Doubly nested', '/Metadata/Series/short_name',
                      ['Metadata_Series', 'short_name']]]

        for description, full_path, expected_key_list in test_args:
            with self.subTest(description):
                self.assertEqual(pydap_attribute_path(full_path),
                                 expected_key_list)

    @patch('pymods.utilities.requests.get')
    def test_get_url_respose(self, mock_requests_get):
        """ Ensure that a successful response will be returned, or one with an
            error status will raise an exception.

        """
        url = 'https://fakesite.org'

        with self.subTest('Succesful response'):
            content = 'CONTENT'
            mock_requests_get.return_value = MockResponse(200, content)
            response = get_url_response(url, self.logger)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, content)

        with self.subTest('Unsuccessful response'):
            with self.assertRaises(HTTPError):
                mock_requests_get.return_value = MockResponse(500, 'ERROR')
                get_url_response(url, self.logger)

    def test_get_xml_namespace(self):
        """ Check that an XML namespace can be retrieved, or if one is absent,
            that a `DmrNamespaceError` is raised.

        """
        with self.subTest('Valid Element'):
            element = ET.fromstring(f'<{self.namespace}Dataset>'
                                    f'</{self.namespace}Dataset>')
            self.assertEqual(get_xml_namespace(element), self.namespace)

        with self.subTest('Non-Dataset Element'):
            element = ET.fromstring(f'<{self.namespace}OtherTag>'
                                    f'</{self.namespace}OtherTag>')
            with self.assertRaises(DmrNamespaceError):
                get_xml_namespace(element)

    def test_get_xml_attribute(self):
        """ Ensure the value of an XML attribute is correctly retrieved, or
            that the default value is returned, where given. This function
            should also cope with non-DAP4 types, by defaulting to a string
            type, before casting the result correctly.

        """
        value = 12.0
        default = 10.0
        variable = ET.fromstring(
            f'<{self.namespace}Int64 name="test_variable">'
            f'  <{self.namespace}Attribute name="valid_attr" type="Float64">'
            f'    <{self.namespace}Value>{value}</{self.namespace}Value>'
            f'  </{self.namespace}Attribute>'
            f'  <{self.namespace}Attribute name="no_type">'
            f'    <{self.namespace}Value>{value}</{self.namespace}Value>'
            f'  </{self.namespace}Attribute>'
            f'  <{self.namespace}Attribute name="no_value" type="String">'
            f'  </{self.namespace}Attribute>'
            f'  <{self.namespace}Attribute name="bad_type" type="Random64">'
            f'    <{self.namespace}Value>{value}</{self.namespace}Value>'
            f'  </{self.namespace}Attribute>'
            f'</{self.namespace}Int64>'
        )

        test_args = [['Element with Float64 attribute', 'valid_attr', value, np.float64],
                     ['Absent Attribute uses default', 'missing_attr', default, type(default)],
                     ['Attribute omitting type property', 'no_type', '12.0', str],
                     ['Absent Value tag uses default', 'no_value', default, type(default)],
                     ['Unexpected type property', 'bad_type', '12.0', str]]

        for description, attr_name, expected_value, expected_type in test_args:
            with self.subTest(description):
                attribute_value = get_xml_attribute(variable, attr_name,
                                                    self.namespace, default)

                self.assertIsInstance(attribute_value, expected_type)
                self.assertEqual(attribute_value, expected_value)

    def test_create_netrc(self):
        """ Ensure a .netrc file can be succesfully created, or if either the
            EDL_USERNAME or EDL_PASSWORD environment variable is absent, that
            an AuthorizationError exception is raised.

        """
        netrc_path = f'{os.path.expanduser("~")}/.netrc'
        username_only = {'EDL_USERNAME': 'sride'}
        password_only = {'EDL_PASSWORD': 'STS-7'}
        full_dictionary = {'EDL_USERNAME': 'sride', 'EDL_PASSWORD': 'STS-7'}

        if os.path.exists(netrc_path):
            os.remove(netrc_path)

        with patch.dict(os.environ, full_dictionary):
            with self.subTest('Successfully creates a .netrc file'):
                create_netrc_file(self.logger)

                with open(netrc_path) as file_handler:
                    written_content = file_handler.readlines()

                self.assertEqual(written_content,
                                 ['machine urs.earthdata.nasa.gov\n',
                                  '    login sride\n',
                                  '    password STS-7\n',
                                  '\n',
                                  'machine uat.urs.earthdata.nasa.gov\n',
                                  '    login sride\n',
                                  '    password STS-7\n',
                                  '\n',
                                  'machine sit.urs.earthdata.nasa.gov\n',
                                  '    login sride\n',
                                  '    password STS-7\n',
                                  '\n'])

        if os.path.exists(netrc_path):
            os.remove(netrc_path)

        test_args = [['Missing EDL_USERNAME raises exception', username_only],
                     ['Missing EDL_PASSWORD raises exception', password_only]]

        for description, environment_variables in test_args:
            with patch.dict(os.environ, environment_variables):
                with self.subTest(description):
                    with self.assertRaises(AuthorizationError):
                        create_netrc_file(self.logger)
                        self.assertFalse(os.path.exists(netrc_path))
