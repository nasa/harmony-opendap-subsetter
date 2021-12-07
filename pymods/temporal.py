""" This module includes functions that support temporal subsets.
    Currently, this includes only geographically gridded data.

    Using prefetched dimension variables, in full, from the OPeNDAP
    server, and the index ranges that correspond to the regions of these
    variables within the specified temporal range are identified. These index
    ranges are then returned to the `pymods.subset.subset_granule` function to
    be combined with any other index ranges (e.g., spatial).

"""
from typing import List, Set

from netCDF4 import Dataset
from varinfo import VarInfoFromDmr, VariableFromDmr
import datetime as dt
from pymods.dimension_utilities import (get_dimension_index_range, IndexRanges)
from dateutil.parser import parse as parse_datetime
from harmony.util import HarmonyException


def get_temporal_index_ranges(required_variables: Set[str],
                                varinfo: VarInfoFromDmr, dimensions_path: str,
                                temporal_range: List[str]) -> IndexRanges:
    """ Iterate through the temporal dimension and extract the indices that
        correspond to the minimum and maximum extents in that dimension. 

        The return value from this function is a dictionary that contains the
        index ranges for the time dimension, such as:

        index_range = {'/time': [1, 13]}

    """
    temporal_range[0] = parse_datetime(temporal_range[0])
    temporal_range[1] = parse_datetime(temporal_range[1])
    index_ranges = {}
    temporal_dimensions = varinfo.get_temporal_dimensions(required_variables)

    units_day = {"day","days","d"}
    units_hour = {"hour","hours","hr","h"}
    units_min = {"minutes","minute","min","mins"}
    units_second = {"second","seconds","sec","secs","s"}

    with Dataset(dimensions_path, 'r') as dimensions_file:
        for dimension in temporal_dimensions:
            var = varinfo.get_variable(dimension)
            units_time = var.get_attribute_value('units')
            ref_time = units_time.split(" since ")[-1].split(" ")
            unit = units_time.split(" since ")[0]
            try:
                if unit in units_day:
                    delta = dt.timedelta(days=1)
                elif unit in units_hour:
                    delta = dt.timedelta(hours=1)
                elif unit in units_min:
                    delta = dt.timedelta(minutes=1)
                elif unit in units_second:
                    delta = dt.timedelta(seconds=1)
            except Exception as exception:
                raise HarmonyException('Subsetter failed with error: ' + str(exception)) from exception

            if (len(ref_time)<2):
                ref_time_hms = "00:00:00"
            else:
                ref_time_hms = ref_time[1]
            time_ref = dt.datetime.fromisoformat(ref_time[0]+"T"+ref_time_hms)
            minimum_extent = (temporal_range[0]- time_ref)/delta
            maximum_extent = (temporal_range[1]- time_ref)/delta
            #print(minimum_extent, maximum_extent)

            index_ranges[dimension] = get_dimension_index_range(
                dimensions_file[dimension][:], minimum_extent, maximum_extent
                )

    return index_ranges
