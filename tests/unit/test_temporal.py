from datetime import datetime, timedelta, timezone
from gc import freeze
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import ANY, patch

import numpy as np
from freezegun import freeze_time
from harmony_service_lib.message import Message
from netCDF4 import Dataset
from numpy.testing import assert_array_equal
from varinfo import VarInfoFromDmr

from hoss.exceptions import UnsupportedTemporalUnits
from hoss.temporal import (
    DEFAULT_TIME_START,
    default_time_end,
    get_datetime_with_timezone,
    get_temporal_index_ranges,
    get_time_ref,
)


class TestTemporal(TestCase):
    """A class for testing functions in the hoss.spatial module."""

    @classmethod
    def setUpClass(cls):
        cls.varinfo = VarInfoFromDmr(
            'tests/data/M2T1NXSLV_example.dmr',
            config_file='tests/data/test_subsetter_config.json',
        )
        cls.test_dir = 'tests/output'

    def setUp(self):
        self.test_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.test_dir)

    def test_get_temporal_index_ranges(self):
        """Ensure that correct temporal index ranges can be calculated."""
        test_file_name = f'{self.test_dir}/test.nc'
        harmony_message = Message(
            {'temporal': {'start': '2021-01-10T01:30:00', 'end': '2021-01-10T05:30:00'}}
        )

        with Dataset(test_file_name, 'w', format='NETCDF4') as test_file:
            test_file.createDimension('time', size=24)

            test_file.createVariable('time', int, dimensions=('time',))
            test_file['time'][:] = np.linspace(0, 1380, 24)
            test_file['time'].setncatts({'units': 'minutes since 2021-01-10 00:30:00'})

        with self.subTest('Time dimension, halfway between the whole hours'):
            self.assertDictEqual(
                get_temporal_index_ranges(
                    {'/time'}, self.varinfo, test_file_name, harmony_message
                ),
                {'/time': (1, 5)},
            )

    @patch('hoss.temporal.get_dimension_index_range')
    def test_get_temporal_index_ranges_bounds(self, mock_get_dimension_index_range):
        """Ensure that bounds are correctly extracted and used as an argument
        for the `get_dimension_index_range` utility function if they are
        present in the prefetch file.

        The GPM IMERG prefetch data are for a granule with a temporal range
        of 2020-01-01T12:00:00 to 2020-01-01T12:30:00.

        """
        mock_get_dimension_index_range.return_value = (1, 2)
        gpm_varinfo = VarInfoFromDmr('tests/data/GPM_3IMERGHH_example.dmr')
        gpm_prefetch_path = 'tests/data/GPM_3IMERGHH_prefetch.nc4'

        harmony_message = Message(
            {'temporal': {'start': '2020-01-01T12:15:00', 'end': '2020-01-01T12:45:00'}}
        )

        self.assertDictEqual(
            get_temporal_index_ranges(
                {'/Grid/time'}, gpm_varinfo, gpm_prefetch_path, harmony_message
            ),
            {'/Grid/time': (1, 2)},
        )
        mock_get_dimension_index_range.assert_called_once_with(
            ANY, 1577880900.0, 1577882700, bounds_values=ANY
        )

        with Dataset(gpm_prefetch_path) as prefetch:
            assert_array_equal(
                mock_get_dimension_index_range.call_args_list[0][0][0],
                prefetch['/Grid/time'][:],
            )
            assert_array_equal(
                mock_get_dimension_index_range.call_args_list[0][1]['bounds_values'],
                prefetch['/Grid/time_bnds'][:],
            )

    def test_get_time_ref(self):
        """Ensure the 'units' attribute tells the correct time_ref and
        time_delta

        """
        expected_datetime = datetime(2021, 12, 8, 0, 30, tzinfo=timezone.utc)

        with self.subTest('units of minutes'):
            self.assertEqual(
                get_time_ref('minutes since 2021-12-08 00:30:00'),
                (expected_datetime, timedelta(minutes=1)),
            )

        with self.subTest('Units of seconds'):
            self.assertEqual(
                get_time_ref('seconds since 2021-12-08 00:30:00'),
                (expected_datetime, timedelta(seconds=1)),
            )

        with self.subTest('Units of hours'):
            self.assertEqual(
                get_time_ref('hours since 2021-12-08 00:30:00'),
                (expected_datetime, timedelta(hours=1)),
            )

        with self.subTest('Units of days'):
            self.assertEqual(
                get_time_ref('days since 2021-12-08 00:30:00'),
                (expected_datetime, timedelta(days=1)),
            )

        with self.subTest('Unrecognised unit'):
            with self.assertRaises(UnsupportedTemporalUnits):
                get_time_ref('fortnights since 2021-12-08 00:30:00')

    @freeze_time("2025-10-21 16:15:00 UTC")
    def test_get_datetime_with_timezone(self):
        """Ensure the string is parsed to datetime with timezone."""
        expected_datetime = datetime(2021, 12, 8, 0, 30, tzinfo=timezone.utc)

        with self.subTest('with space'):
            self.assertEqual(
                get_datetime_with_timezone('2021-12-08 00:30:00'), expected_datetime
            )

        with self.subTest('no space'):
            self.assertEqual(
                get_datetime_with_timezone('2021-12-08T00:30:00'), expected_datetime
            )

        with self.subTest('no space with trailing Z'):
            self.assertEqual(
                get_datetime_with_timezone('2021-12-08T00:30:00Z'), expected_datetime
            )

        with self.subTest('space with trailing Z'):
            self.assertEqual(
                get_datetime_with_timezone('2021-12-08 00:30:00Z'), expected_datetime
            )

        with self.subTest('ISO with fractional seconds Z'):
            self.assertEqual(
                get_datetime_with_timezone('2021-12-08T00:30:00.000Z'),
                expected_datetime,
            )

        with self.subTest('DEFAULT_TIME_START'):
            expected_start_time = datetime(1, 1, 1, 0, 0, tzinfo=timezone.utc)
            self.assertEqual(
                get_datetime_with_timezone(DEFAULT_TIME_START), expected_start_time
            )

        with self.subTest('default_end_time()'):
            expected_stop_time = datetime(2025, 10, 21, 16, 15, tzinfo=timezone.utc)
            self.assertEqual(
                get_datetime_with_timezone(default_time_end()), expected_stop_time
            )
