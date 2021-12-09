from logging import getLogger
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import Mock
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_datetime

from netCDF4 import Dataset
import numpy as np

from varinfo import VarInfoFromDmr
from harmony.util import HarmonyException
from pymods.temporal import (get_temporal_index_ranges,
                             get_time_ref,
                             get_datetime_with_timezone)


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
        """ Ensure that correct temporal index ranges can be calculated.

        """
        test_file_name = f'{self.test_dir}/test.nc'
        temporal_range = ['2021-01-10T01:30:00', '2021-01-10T05:30:00']

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
        """ Ensure the 'units' attribute tells the correct time_ref and time_delta

        """

        self.assertEqual(
                get_time_ref('minutes since 2021-12-08 00:30:00'),
                (datetime(2021, 12, 8, 0, 30, tzinfo=timezone.utc),
                 timedelta(minutes=1)))

        with self.subTest('Units of seconds'):
            self.assertEqual(get_time_ref('seconds since 2021-12-08 00:30:00'),
                             (datetime(2021, 12, 8, 0, 30, tzinfo=timezone.utc),
                             timedelta(seconds=1)))

        with self.subTest('Units of hours'):
            self.assertEqual(get_time_ref('hours since 2021-12-08 00:30:00'),
                             (datetime(2021, 12, 8, 0, 30, tzinfo=timezone.utc),
                             timedelta(hours=1)))

        with self.subTest('Units of days'):
            self.assertEqual(get_time_ref('days since 2021-12-08 00:30:00'),
                             (datetime(2021, 12, 8, 0, 30, tzinfo=timezone.utc),
                             timedelta(days=1)))

        with self.subTest('Unrecognised unit'):
            with self.assertRaises(HarmonyException):
                get_time_ref('fortnights since 2021-12-08 00:30:00')

    def test_get_datetime_with_timezone(self):
        """ Ensure the string is parsed to datetime with timezone

        """

        with self.subTest('with space'):
            self.assertEqual(
                    get_datetime_with_timezone('2021-12-08 00:30:00'),
                    datetime(2021, 12, 8, 0, 30, tzinfo=timezone.utc))

        with self.subTest('no space'):
            self.assertEqual(
                    get_datetime_with_timezone('2021-12-08T00:30:00'),
                    datetime(2021, 12, 8, 0, 30, tzinfo=timezone.utc))
        
        with self.subTest('no space with trailing Z'):
            self.assertEqual(
                    get_datetime_with_timezone('2021-12-08T00:30:00Z'),
                    datetime(2021, 12, 8, 0, 30, tzinfo=timezone.utc))
        
        with self.subTest('space with trailing Z'):
            self.assertEqual(
                    get_datetime_with_timezone('2021-12-08 00:30:00Z'),
                    datetime(2021, 12, 8, 0, 30, tzinfo=timezone.utc))
