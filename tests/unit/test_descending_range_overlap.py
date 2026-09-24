"""Regression tests for overlapping requests on descending dimensions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from harmony_service_lib.exceptions import NoDataException
from harmony_service_lib.message import Message
from netCDF4 import Dataset
from varinfo import VarInfoFromDmr

from hoss.dimension_utilities import (
    add_index_range,
    get_dimension_index_range,
    get_dimension_indices_from_values,
    get_requested_index_ranges,
)
from hoss.exceptions import InvalidRequestedRange


class TestDescendingRangeOverlap(TestCase):
    """A reversed axis must retain the same overlapping coordinate samples."""

    def test_clipped_ranges_in_both_directions(self):
        """Clip overlapping endpoints using the existing interpolation rules."""
        cases = [(-5, 10, (0, 1)), (10, 25, (1, 2)), (-5, 25, (0, 2))]
        for dtype in (np.float32, np.float64):
            for descending in (False, True):
                values = np.ma.array([0, 10, 20], dtype=dtype)
                if descending:
                    values = values[::-1]
                original = values.copy()
                for low, high, ascending_indices in cases:
                    with self.subTest(
                        dtype=dtype, descending=descending, low=low, high=high
                    ):
                        expected = ascending_indices
                        if descending:
                            expected = (2 - expected[1], 2 - expected[0])
                        self.assertEqual(
                            get_dimension_index_range(values, low, high), expected
                        )
                        extents = (high, low) if descending else (low, high)
                        self.assertEqual(
                            get_dimension_indices_from_values(values, *extents),
                            expected,
                        )
                        np.testing.assert_array_equal(values, original)

    def test_disjoint_ranges_remain_rejected(self):
        """A range wholly outside either end must not produce a data slice."""
        for descending in (False, True):
            values = np.ma.array([20, 10, 0] if descending else [0, 10, 20])
            for low, high in ((-20, -10), (30, 40), (-5, -5), (25, 25)):
                with self.subTest(descending=descending, low=low, high=high):
                    with self.assertRaises(InvalidRequestedRange):
                        get_dimension_index_range(values, low, high)

    def test_interior_and_optional_endpoints_are_unchanged(self):
        """Keep point rounding and open-ended requests in both directions."""
        for descending in (False, True):
            values = np.ma.array([20.0, 10.0, 0.0] if descending else [0.0, 10.0, 20.0])
            for low, high, indices in (
                (0, 20, (0, 2)),
                (10, 10, (1, 1)),
                (5, 5, (0, 1)),
                (None, 10, (0, 1)),
                (10, None, (1, 2)),
                (None, None, (0, 2)),
            ):
                with self.subTest(descending=descending, low=low, high=high):
                    expected = (
                        (2 - indices[1], 2 - indices[0]) if descending else indices
                    )
                    self.assertEqual(
                        get_dimension_index_range(values, low, high), expected
                    )

    def test_named_dimension_request_reads_real_netcdf(self):
        """Exercise DMR parsing, NetCDF reads and OPeNDAP slice construction."""
        dmr = """<Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#" name="sample">
          <Dimension name="level" size="3"/>
          <Float64 name="level"><Dim name="/level"/>
            <Attribute name="units" type="String"><Value value="hPa"/></Attribute>
          </Float64>
          <Float64 name="temperature"><Dim name="/level"/></Float64>
        </Dataset>"""
        with TemporaryDirectory() as directory:
            dmr_path = Path(directory) / 'sample.dmr'
            dmr_path.write_text(dmr, encoding='utf-8')
            varinfo = VarInfoFromDmr(str(dmr_path))
            for descending in (False, True):
                path = Path(directory) / f'{descending}.nc4'
                values = np.array(
                    [20.0, 10.0, 0.0] if descending else [0.0, 10.0, 20.0]
                )
                with Dataset(path, 'w') as dataset:
                    dataset.createDimension('level', 3)
                    dataset.createVariable('level', 'f8', ('level',))[:] = values
                    dataset.createVariable('temperature', 'f8', ('level',))[:] = (
                        values + 273
                    )
                for low, high in ((-5, 10), (10, 25), (-5, 25)):
                    with self.subTest(descending=descending, low=low, high=high):
                        message = Message(
                            {
                                'subset': {
                                    'dimensions': [
                                        {'name': '/level', 'min': low, 'max': high}
                                    ]
                                }
                            }
                        )
                        indices = get_requested_index_ranges(
                            {'/temperature'}, varinfo, str(path), message
                        )
                        wanted = np.flatnonzero((values >= low) & (values <= high))
                        expected = (int(wanted[0]), int(wanted[-1]))
                        self.assertEqual(indices, {'/level': expected})
                        self.assertEqual(
                            add_index_range('/temperature', varinfo, indices),
                            f'/temperature[{expected[0]}:{expected[1]}]',
                        )
                        with Dataset(path) as dataset:
                            np.testing.assert_array_equal(
                                dataset['temperature'][expected[0] : expected[1] + 1],
                                (values + 273)[wanted],
                            )
                message = Message(
                    {
                        'subset': {
                            'dimensions': [{'name': '/level', 'min': 30, 'max': 40}]
                        }
                    }
                )
                with self.assertRaises(NoDataException):
                    get_requested_index_ranges(
                        {'/temperature'}, varinfo, str(path), message
                    )
