"""This module includes functions that support temporal subsets.

Using prefetched dimension variables, in full, from the OPeNDAP
server, and the index ranges that correspond to the regions of these
variables within the specified temporal range are identified. These index
ranges are then returned to the `hoss.subset.subset_granule` function to
be combined with any other index ranges (e.g., spatial).

"""

from datetime import datetime, timedelta, timezone
from typing import List, Set

from dateutil import tz
from dateutil.parser import parse as parse_datetime
from harmony_service_lib.exceptions import NoDataException
from harmony_service_lib.message import Message
from netCDF4 import Dataset
from varinfo import VarInfoFromDmr

from hoss.dimension_utilities import (
    IndexRanges,
    check_range_exception,
    get_dimension_bounds,
    get_dimension_index_range,
)
from hoss.exceptions import InvalidRequestedRange, UnsupportedTemporalUnits

units_day = {'day', 'days', 'd'}
units_hour = {'hour', 'hours', 'hr', 'h'}
units_min = {'minutes', 'minute', 'min', 'mins'}
units_second = {'second', 'seconds', 'sec', 'secs', 's'}


DEFAULT_TIME_START = '0001-01-01T00:00:00.000Z'


def default_time_end():
    """Returns the current UTC time to be used as a default ending time."""
    return datetime.now(tz=tz.UTC).strftime('%Y-%m-%dT%H:%M:%S.000Z')


def get_temporal_index_ranges(
    required_variables: Set[str],
    varinfo: VarInfoFromDmr,
    dimensions_path: str,
    harmony_message: Message,
) -> IndexRanges:
    """Iterate through the temporal dimension and extract the indices that
    correspond to the minimum and maximum extents in that dimension.

    The return value from this function is a dictionary that contains the
    index ranges for the time dimension, such as:

    index_range = {'/time': [1, 5]}

    """
    index_ranges = {}
    failed_variables = []
    temporal_dimensions = varinfo.get_temporal_dimensions(required_variables)

    time_start = get_datetime_with_timezone(
        harmony_message.temporal.start or DEFAULT_TIME_START
    )
    time_end = get_datetime_with_timezone(
        harmony_message.temporal.end or default_time_end()
    )

    with Dataset(dimensions_path, 'r') as dimensions_file:
        for dimension in temporal_dimensions:
            try:
                time_variable = varinfo.get_variable(dimension)
                time_ref, time_delta = get_time_ref(
                    time_variable.get_attribute_value('units')
                )

                # Convert the Harmony message start and end datetime values into
                # integer or floating point values (e.g., a number of seconds since
                # 1970-01-01) using the variable epoch and unit.
                minimum_extent = (time_start - time_ref) / time_delta
                maximum_extent = (time_end - time_ref) / time_delta

                index_ranges[dimension] = get_dimension_index_range(
                    dimensions_file[dimension][:],
                    minimum_extent,
                    maximum_extent,
                    bounds_values=get_dimension_bounds(
                        dimension, varinfo, dimensions_file
                    ),
                )
            # pylint: disable=duplicate-code
            except InvalidRequestedRange:
                check_range_exception(
                    required_variables,
                    failed_variables,
                    dimension,
                    varinfo,
                )
            # pylint: enable=duplicate-code

        if failed_variables:
            raise NoDataException(
                f'Temporal range request outside supported dimension range for {failed_variables}'
            )

    return index_ranges


def get_datetime_with_timezone(timestring: str) -> datetime:
    """function to parse string to datetime, and ensure datetime is timezone
    "aware". If a timezone is not supplied, it is assumed to be UTC.

    """

    parsed_datetime = parse_datetime(timestring)

    if parsed_datetime.tzinfo is None:
        parsed_datetime = parsed_datetime.replace(tzinfo=timezone.utc)

    return parsed_datetime


def get_time_ref(units_time: str) -> List[datetime]:
    """Retrieve the reference time (epoch) and time step size."""
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
