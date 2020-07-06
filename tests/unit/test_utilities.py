from logging import Logger
from unittest import TestCase
from unittest.mock import patch

from harmony.message import Granule
from pymods.utilities import (get_file_mimetype,
                              pydap_attribute_path, recursive_get, get_url_response)

class TestUtilities(TestCase):
    """ A class for testing functions in the pymods.utilities module. """

    @classmethod
    def setUpClass(cls):
        cls.granule = Granule({'url': '/home/tests/data/africa.nc'})
        cls.granule.local_filename = cls.granule.url
        cls.source_url = 'https://opendap.uat.earthdata.nasa.gov/providers/EEDTEST/collections/ATLAS-ICESat-2%20L2A%20Global%20Geolocated%20Photon%20Data%20V003/granules/EEDTEST-ATL03-003-ATL03_20181228015957.dmr'

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

    def test_url_respose(self):
        """ Ensure a response can be retrieved for a valid source file url"""

        response = get_url_response(self.source_url, self.logger)
        if (response != None):
            self.assertEqual(response.status_code, 200)



