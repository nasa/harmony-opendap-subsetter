from logging import Logger
from unittest import TestCase
import json

from harmony.message import Message

from pymods.subset import subset_granule


class TestSubset(TestCase):
    """ Test the module that performs subsetting on a single granule. """

    @classmethod
    def setUpClass(cls):
        #cls.granule = Granule({'url': '/home/tests/data/africa.nc', 'id': 'G1233860471-EEDTEST',
        #                       'collection': 'C1233860183-EEDTEST', 'variables': ['/gt1r/geolocation/segment_id']})
        cls.message_content = ({'sources': [{'collection': 'C1233860183-EEDTEST',
                                           'variables': [{'id': 'V1234834148-EEDTEST',
                                                          'name': 'geoid',
                                                          'fullPath': 'gtr1/geophys_corr/geoid'}],
                                           'granules': [{'id': 'G1233860471-EEDTEST',
                                                         'url': '/home/tests/data/africa.nc'}]
                                            }]})
        cls.message = Message(json.dumps(cls.message_content))

    def setUp(self):
        self.logger = Logger('tests')

    def test_subset_granule(self):
        """ This is a placeholder test that should be updated when the proper
            functionality for `subset_granule` is implemented.

        """
        granule = self.message.granules[0]
        output_path = subset_granule(granule, self.logger)
        self.assertEqual(output_path, '/path/to/subsetting/output.nc')
