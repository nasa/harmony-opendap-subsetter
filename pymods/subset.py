""" The module contains the main functions to perform variable subsetting on a
    single granule file. This should all be wrapped by the `subset_granule`
    function, which is called from the `HarmonyAdapter` class.

"""
from logging import Logger
from typing import List, Optional, Set
from datetime import datetime

from harmony.message import Variable as HarmonyVariable
from harmony.util import Config
from netCDF4 import Dataset
from numpy.ma import masked
from varinfo import VarInfoFromDmr

from pymods.bbox_utilities import (BBox, get_shape_file_geojson,
                                   get_geographic_bbox)
from pymods.dimension_utilities import (add_index_range, get_fill_slice,
                                        IndexRanges, is_index_subset,
                                        prefetch_dimension_variables)
from pymods.spatial import get_geographic_index_ranges
from pymods.temporal import get_temporal_index_ranges
from pymods.utilities import (download_url, format_variable_set_string,
                              get_opendap_nc4)


def subset_granule(opendap_url: str, variables: List[HarmonyVariable],
                   output_dir: str, logger: Logger, access_token: str = None,
                   config: Config = None, bounding_box: BBox = None,
                   shape_file_path: str = None,
                   temporal_range: List[datetime] = None) -> str:
    """ This function is the main business logic for retrieving a variable
        and/or spatial subset from OPeNDAP.

        Variable dependencies are extracted from an `varinfo.VarInfoFromDmr`
        instance that is based on the `.dmr` file for the granule as obtained
        from OPeNDAP. The full set of returned variables will include those
        requested by the end-user, and additional variables required to support
        those requested (e.g., grid dimension variables or CF-Convention
        metadata references).

        The optional `bounding_box` argument can be supplied for geographically
        gridded data. In this case dimension variables will first be retrieved
        in a "prefetch" request to OPeNDAP. Then the bounding-box extents are
        converted to index-ranges.

        Once the required variables, and index-ranges if needed, are derived,
        a request is made to OPeNDAP to retrieve only the requested data.

        Future work: When temporal subsetting is to be added, the temporal
        index ranges should be extracted in a new module and added to the
        existing `index_ranges` cache.

    """
    # Determine if index range subsetting will be required:
    request_is_index_subset = is_index_subset(bounding_box, shape_file_path,
                                              temporal_range)

    # Produce a map of variable dependencies with `sds-varinfo` and the `.dmr`.
    varinfo = get_varinfo(opendap_url, output_dir, logger, access_token, config)

    requested_variables = get_requested_variables(varinfo, variables,
                                                  request_is_index_subset)
    logger.info('Requested variables: '
                f'{format_variable_set_string(requested_variables)}')

    # Obtain a list of all variables for the subset, including those used as
    # references by the requested variables.
    required_variables = varinfo.get_required_variables(requested_variables)
    logger.info('All required variables: '
                f'{format_variable_set_string(required_variables)}')

    # Define a catch to store all dimension index ranges (spatial, temporal):
    index_ranges = {}

    # If there is no bounding box, but there is a shape-file, calculate a
    # bounding box to encapsulate the GeoJSON shape:
    if bounding_box is None and shape_file_path is not None:
        geojson_content = get_shape_file_geojson(shape_file_path)
        bounding_box = get_geographic_bbox(geojson_content)

    if request_is_index_subset:
        # Prefetch all dimension variables in full:
        dimensions_path = prefetch_dimension_variables(opendap_url, varinfo,
                                                       required_variables,
                                                       output_dir, logger,
                                                       access_token, config)

        if bounding_box is not None:
            # Update `index_ranges` cache with ranges for geographic grid-dimension
            # variables. This will convert information from the bounding box to
            # array indices for each geographic grid-dimension.
            index_ranges.update(get_geographic_index_ranges(required_variables,
                                                            varinfo,
                                                            dimensions_path,
                                                            bounding_box))
        if temporal_range is not None:
            # Update `index_ranges` cache with ranges for temporal
            # variables. This will convert information from the temporal range
            # to array indices for each temporal dimension.
            index_ranges.update(get_temporal_index_ranges(required_variables,
                                                          varinfo,
                                                          dimensions_path,
                                                          temporal_range))

    # Add any range indices to variable names for DAP4 constraint expression.
    variables_with_ranges = set(
        add_index_range(variable, varinfo, index_ranges)
        for variable in required_variables
    )

    # Retrieve OPeNDAP data including only the specified variables in the
    # specified ranges.
    output_path = get_opendap_nc4(opendap_url, variables_with_ranges,
                                  output_dir, logger, access_token, config)

    # Fill the data outside the requested ranges for variables that cross a
    # dimensional discontinuity (for example longitude and the anti-meridian).
    fill_variables(output_path, varinfo, required_variables, index_ranges)

    return output_path


def get_varinfo(opendap_url: str, output_dir: str, logger: Logger,
                access_token: str, config: Config) -> str:
    """ Retrieve the `.dmr` from OPeNDAP and use `sds-varinfo` to populate a
        representation of the granule that maps dependencies between variables.

    """
    dmr_path = download_url(f'{opendap_url}.dmr.xml', output_dir, logger,
                            access_token=access_token, config=config)
    return VarInfoFromDmr(dmr_path, logger,
                          config_file='pymods/var_subsetter_config.yml')


def get_requested_variables(varinfo: VarInfoFromDmr,
                            variables: List[HarmonyVariable],
                            request_is_index_subset: bool) -> Set[str]:
    """ Iterate through all requested variables from the Harmony message and
        extract their full paths.

        If index range subsetting is required, but no variables are specified
        (e.g., all variables are requested) then the requested variables should
        be set to all variables (science and non-science), so that index-range
        subsets can be specified in a DAP4 constraint expression.

        NOTE: When adding temporal subsetting, a condition will need to be
        added, so the check is: if (spatial or temporal) and not variables

    """
    requested_variables = set(variable.fullPath
                              if variable.fullPath.startswith('/')
                              else f'/{variable.fullPath}'
                              for variable in variables)

    if request_is_index_subset and len(requested_variables) == 0:
        requested_variables = varinfo.get_science_variables().union(
            varinfo.get_metadata_variables()
        )

    return requested_variables


def fill_variables(output_path: str, varinfo: VarInfoFromDmr,
                   required_variables: Set[str],
                   index_ranges: IndexRanges) -> None:
    """ Check the index ranges for all dimension variables. If the minimum
        index is greater than the maximum index in the subset range, then the
        requested dimension range crossed an edge of the grid (e.g. longitude),
        and must be filled in between those values.

        Note - longitude variables themselves will not be filled, to ensure
        valid grid coordinates at all points of the science variables.

    """
    fill_ranges = {dimension: index_range
                   for dimension, index_range
                   in index_ranges.items()
                   if index_range[0] > index_range[1]}

    dimensions_to_fill = set(fill_ranges)

    if len(dimensions_to_fill) > 0:
        with Dataset(output_path, 'a', format='NETCDF4') as output_dataset:
            for variable_path in required_variables:
                fill_variable(output_dataset, fill_ranges, varinfo,
                              variable_path, dimensions_to_fill)


def fill_variable(output_dataset: Dataset, fill_ranges: IndexRanges,
                  varinfo: VarInfoFromDmr, variable_path: str,
                  dimensions_to_fill: Set[str]) -> None:
    """ Check if the variable has dimensions that require filling. If so,
        and if the variable is not the longitude itself, fill the data outside
        of the requested dimension range using the `numpy.ma.masked` constant.
        The dimension variables should not be filled to ensure there are valid
        grid-dimension values for all pixels in the grid.

        Conditions for filling:

        * Variable is not the longitude dimension (currently the only dimension
          we expect to cross a grid edge).
        * Variable has at least one grid-dimension that crosses a grid edge.

    """
    variable = varinfo.get_variable(variable_path)

    if (
            not variable.is_longitude()
            and len(dimensions_to_fill.intersection(variable.dimensions)) > 0
    ):
        fill_index_tuple = tuple(
            get_fill_slice(dimension, fill_ranges)
            for dimension in variable.dimensions
        )

        output_dataset[variable_path][fill_index_tuple] = masked
