""" This module includes functions that support temporal subsets.
    Currently, this includes only geographically gridded data.

    Using prefetched dimension variables, in full, from the OPeNDAP
    server, and the index ranges that correspond to the regions of these
    variables within the specified temporal range are identified. These index
    ranges are then returned to the `pymods.subset.subset_granule` function to
    be combined with any other index ranges (e.g., spatial).

"""
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_datetime
from typing import List, Set

from netCDF4 import Dataset
from varinfo import VarInfoFromDmr

from pymods.dimension_utilities import get_dimension_index_range, IndexRanges
from pymods.exceptions import UnsupportedTemporalUnits


units_day = {'day', 'days', 'd'}
units_hour = {'hour', 'hours', 'hr', 'h'}
units_min = {'minutes', 'minute', 'min', 'mins'}
units_second = {'second', 'seconds', 'sec', 'secs', 's'}


def get_datetime_with_timezone(timestring: str) -> datetime:
    """ function to parse string to datetime, and ensure datetime is timezone
        "aware". If a timezone is not supplied, it is assumed to be UTC.

    """

    parsed_datetime = parse_datetime(timestring)

    if parsed_datetime.tzinfo is None:
        parsed_datetime = parsed_datetime.replace(tzinfo=timezone.utc)

    return parsed_datetime


def get_time_ref(units_time: str) -> List[datetime]:
    """ retrieve the reference time and time step size

    """
    unit, epoch_str = units_time.split(' since ')
    ref_time = get_datetime_with_timezone(epoch_str)

    if unit in units_day:
        time_delta = timedelta(days=1)
    elif unit in units_hour:
        time_delta = timedelta(hours=1)
    elif unit in units_min:
        time_delta = timedelta(minutes=1)
    elif unit in units_second:
        time_delta = timedelta(seconds=1)
    else:
        raise UnsupportedTemporalUnits(unit)

    return (ref_time, time_delta)


def get_temporal_index_ranges(required_variables: Set[str],
                              varinfo: VarInfoFromDmr, dimensions_path: str,
                              temporal_range: List[str]) -> IndexRanges:
    """ Iterate through the temporal dimension and extract the indices that
        correspond to the minimum and maximum extents in that dimension.

        The return value from this function is a dictionary that contains the
        index ranges for the time dimension, such as:

        index_range = {'/time': [1, 5]}

    """
    index_ranges = {}
    temporal_dimensions = varinfo.get_temporal_dimensions(required_variables)

    time_start = get_datetime_with_timezone(temporal_range[0])
    time_end = get_datetime_with_timezone(temporal_range[1])

    with Dataset(dimensions_path, 'r') as dimensions_file:
        for dimension in temporal_dimensions:
            time_variable = varinfo.get_variable(dimension)
            units_time = time_variable.get_attribute_value('units')
            time_ref, time_delta = get_time_ref(units_time)

            minimum_extent = (time_start - time_ref)/time_delta
            maximum_extent = (time_end - time_ref)/time_delta

            index_ranges[dimension] = get_dimension_index_range(
                dimensions_file[dimension][:], minimum_extent, maximum_extent
            )

    return index_ranges
