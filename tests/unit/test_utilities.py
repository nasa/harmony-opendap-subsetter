import json
from logging import Logger
from unittest import TestCase

from unittest.mock import patch, Mock
import os

from harmony.message import Granule, Message
from requests.exceptions import HTTPError

from pymods.exceptions import AuthorizationError
from pymods.subset import subset_granule
from pymods.utilities import (get_file_mimetype,
                              pydap_attribute_path, recursive_get, get_url_response)
import xml.etree.ElementTree as ET

from harmony.message import Granule
import numpy as np

from pymods.exceptions import DmrNamespaceError
from pymods.utilities import (get_file_mimetype, get_xml_attribute,
                              get_xml_namespace, pydap_attribute_path,
                              recursive_get)


def generate_response(status=200,
                      content=b'CONTENT',
                      json_data=None,
                      raise_for_status=None,
                      url='https://fakesite.org'):
    mock_resp = Mock()
    mock_resp.raise_for_status = Mock()
    if (url == 'https://fakesite.org'):
        mock_resp.return_value = content
    if raise_for_status:
        mock_resp.raise_for_status.side_effect = raise_for_status
    mock_resp.status_code = status
    mock_resp.content = content
    if json_data:
        mock_resp.json = Mock(return_value=json_data)

    return mock_resp

class TestUtilities(TestCase):
    """ A class for testing functions in the pymods.utilities module. """

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
        cls.granule = Granule({'url': '/home/tests/data/africa.nc'})
        cls.granule.local_filename = cls.granule.url
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

    @patch('pymods.subset.VarInfo')
    @patch('pymods.subset.requests.get')
    @patch.dict(os.environ, {"EDL_USERNAME": "fake_username", "EDL_PASSWORD":"fake_password"})
    def test_subset_granule(self, mock_get, mock_var_info):
        """ Ensure valid request does not raise exception,
            raise appropriate exception otherwise.

        """
        granule = self.message.granules[0]
        granule.local_filename = '/home/tests/data/africa.nc'
        mock_response = generate_response()
        mock_get.return_value = mock_response

        output_path = subset_granule(granule, self.logger)
        mock_get.assert_called_once()
        self.assertIn('africa_subset.nc', output_path)

        with self.subTest('Unauthorized error'):
            with self.assertRaises(HTTPError):
                mock_response = generate_response(status=401,
                                                  raise_for_status=HTTPError(
                                                      "Request cannot be completed with error code 401"))
                mock_get.return_value = mock_response
                subset_granule(granule, self.logger)

        with self.subTest('Service Unavailable'):
            with self.assertRaises(HTTPError):
                mock_response = generate_response(status=500,
                                                  raise_for_status=HTTPError(
                                                      "Request cannot be completed with error code 500"))
                mock_get.return_value = mock_response
                subset_granule(granule, self.logger)

    @patch('pymods.utilities.get_url_response')
    def test_get_url_respose(self, mock_get_url_response):
        """ Ensure that if System variables EDL_USERNAME and EDL_PASSWORD
            are one or both None then message will raised

        """
        url = 'https://fakesite.org'
        with self.assertRaises(AuthorizationError):
            mock_response = generate_response(None,
                                              raise_for_status=AuthorizationError(
                                                  "There are no EDL_USERNAME and EDL_PASSWORD in the system environment"))
            mock_get_url_response.return_value = mock_response
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
