from unittest import TestCase
from unittest.mock import patch

from harmony.message import Granule

from pymods.utilities import get_granule_mimetype, get_token, cmr_query


class TestUtilities(TestCase):
    """ A class for testing functions in the pymods.utilities module. """

    @classmethod
    def setUpClass(cls):
        cls.granule = Granule({'url': '/home/tests/data/africa.nc'})
        cls.granule.local_filename = cls.granule.url

    def test_get_granule_mimetype(self):
        """ Ensure a mimetype can be retrieved for a valid file path or, if
            the mimetype cannot be inferred, that the default output is
            returned.

        """
        with self.subTest('File with MIME type'):
            mimetype = get_granule_mimetype(self.granule)
            self.assertEqual(mimetype, ('application/x-netcdf', None))

        with self.subTest('Default MIME type is returned'):
            with patch('mimetypes.guess_type') as mock_guess_type:
                mock_guess_type.return_value = (None, None)
                mimetype = get_granule_mimetype(self.granule)
                self.assertEqual(mimetype, ('application/octet-stream', None))

    def test_cmr_query(self):
        """ CMR queries returned correct response """
        token = get_token()
        collection_id = 'C1234714691-EEDTEST'
        granule_id = 'G1234718422-EEDTEST'

        with self.subTest('Collection entry title'):
            entry_title = cmr_query('collections', collection_id, 'EntryTitle', token)
            self.assertEqual(entry_title, 'ATLAS-ICESat-2 L2A Global Geolocated Photon Data V003')

        with self.subTest('Granule granuleUR'):
            granule_ur = cmr_query('granules', granule_id, 'GranuleUR', token)
            self.assertEqual(granule_ur, 'EEDTEST-ATL03-003-ATL03_20181228T013120')
