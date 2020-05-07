from logging import Logger
from unittest import TestCase

from harmony.message import Granule

from pymods.subset import subset_granule


class TestSubset(TestCase):
    """ Test the module that performs subsetting on a single granule. """

    @classmethod
    def setUpClass(cls):
        cls.granule = Granule({'url': '/home/tests/data/africa.nc'})

    def setUp(self):
        self.logger = Logger('tests')

    def test_subset_granule(self):
        """ This is a placeholder test that should be updated when the proper
            functionality for `subset_granule` is implemented.

        """
        output_path = subset_granule(self.granule, self.logger)
        self.assertEqual(output_path, '/path/to/subsetting/output.nc')
