from logging import Logger
from unittest import TestCase

from harmony.message import Granule
from pymods.var_info import VarInfo

class TestVarinfo(TestCase):
    """ Test Varinfo. """

    @classmethod
    def setUpClass(cls):
        cls.url = 'http://test.opendap.org/opendap/hyrax/slav/ATL08_20181016124656_02730110_002_01.h5'
        #cls.url = 'http://opendap.uat.earthdata.nasa.gov/providers/EEDTEST/collections/ATLAS-ICESat-2%20L2A%20Global%20Geolocated%20Photon%20Data%20V003/granules/EEDTEST-ATL03-003-ATL03_20181228T013120'

        cls.dataset = dataset = VarInfo(cls.url)

    def test_Varinfo(self):
        """ This is a placeholder test that should be updated when the proper
            functionality for `subset_granule` is implemented.

        """
        #req_vars = self.dataset.get_required_variables({'/gt2r/heights/h_ph'})
        req_vars = self.dataset.get_required_variables({'/gt1r/land_segments/canopy/h_canopy_abs'})
        print(min(req_vars))
        #self.assertEqual(min(req_vars),"/gt2r/heights/delta_time")
        self.assertEqual(min(req_vars), "/gt1r/land_segments/canopy/h_canopy_abs")