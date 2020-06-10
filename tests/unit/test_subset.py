from logging import Logger
from unittest import TestCase
import json

from harmony.message import Message

from pymods.subset import subset_granule


class TestSubset(TestCase):
    """ Test the module that performs subsetting on a single granule. """

    @classmethod
    def setUpClass(cls):
        cls.message_content = ({'sources': [{'collection': 'C1233860183-EEDTEST',
                                             'variables': [{'id': 'V1234834148-EEDTEST',
                                                            'name': 'geoid',
                                                            'fullPath': 'gt1r/geophys_corr/geoid'}],
                                             'granules': [{'id': 'G1233860471-EEDTEST',
                                                           'url': '/home/tests/data/africa.nc'}]
                                             }]})

        cls.message = Message(json.dumps(cls.message_content))

    def setUp(self):
        self.logger = Logger('tests')

    def test_subset_granule(self):
        """ A request with variables specified should return an output
            with requested variables.
        """
        granule = self.message.granules[0]
        output_path = subset_granule(granule, self.logger)
        self.assertEqual(output_path, '/path/to/subsetting/output.nc')
