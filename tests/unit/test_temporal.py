from logging import getLogger
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import Mock
from datetime import datetime, timedelta, timezone

from netCDF4 import Dataset
import numpy as np

from varinfo import VarInfoFromDmr
from pymods.temporal import (get_temporal_index_ranges,
                            get_time_ref)

class TestTemporal(TestCase):
    """ A class for testing functions in the pymods.spatial module. """
    @classmethod
    def setUpClass(cls):
        cls.logger = getLogger('tests')
        cls.varinfo = VarInfoFromDmr('tests/data/M2T1NXSLV.5.12.4%3AMERRA2_400.tavg1_2d_slv_Nx.20210110.nc4.dmr',
                                     cls.logger,
                                     'tests/data/test_subsetter_config.yml')
        cls.test_dir = 'tests/output'

    def setUp(self):
        self.test_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.test_dir)

    def test_get_temporal_index_ranges(self):
        """ Ensure that correct index ranges can be calculated.

        """
        test_file_name = f'{self.test_dir}/test.nc'
        temporal_range = [datetime(2021, 1, 10, 1, 30), datetime(2021, 1, 10, 5, 30)]

        with Dataset(test_file_name, 'w', format='NETCDF4') as test_file:
            test_file.createDimension('time', size=24)

            test_file.createVariable('time', int,
                                     dimensions=('time', ))
            test_file['time'][:] = np.linspace(0, 1380, 24)
            test_file['time'].setncatts({'units': 'minutes since 2021-01-10 00:30:00'})

        with self.subTest('Time dimension, halfway between the whole hours'):
            self.assertDictEqual(
                get_temporal_index_ranges({'/time'}, self.varinfo,
                                            test_file_name, temporal_range),
                {'/time': (1, 5)}
            )
    
    def test_get_time_ref(self):
        """ Ensure that get_time_ref returns the correct time_ref and time_delta
        
        """

        self.assertEqual(
                get_time_ref('minutes since 2021-12-08 00:30:00'),
                (datetime(2021, 12, 8, 0, 30, tzinfo=timezone.utc), 
                timedelta(minutes=1)))
