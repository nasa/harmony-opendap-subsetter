"""The module contains the main functions to perform variable, spatial,
temporal and named-dimension subsetting on a single granule file. This is
wrapped by the `subset_granule` function, which is called from the
`hoss.adapter.HossAdapter` class.

"""

from logging import Logger
from typing import List, Set

from harmony_service_lib.message import Message, Source
from harmony_service_lib.message import Variable as HarmonyVariable
from harmony_service_lib.message_utility import rgetattr
from harmony_service_lib.util import Config
from netCDF4 import Dataset
from numpy.ma import masked
from varinfo import VarInfoFromDmr

from hoss.bbox_utilities import get_request_shape_file
from hoss.dimension_utilities import (
    IndexRanges,
    add_index_range,
    get_fill_slice,
    get_prefetch_variables,
    get_requested_index_ranges,
    is_index_subset,
)
from hoss.spatial import get_spatial_index_ranges
from hoss.temporal import get_temporal_index_ranges
from hoss.utilities import (
    download_url,
    format_variable_set_string,
    get_opendap_nc4,
)


def subset_granule(
    opendap_url: str,
    harmony_source: Source,
    output_dir: str,
    harmony_message: Message,
    logger: Logger,
    config: Config,
) -> str:
    """This function is the main business logic for retrieving a variable,
    spatial, temporal and/or named-dimension subset from OPeNDAP.

    Variable dependencies are extracted from a `varinfo.VarInfoFromDmr`
    instance that is based on the `.dmr` file for the granule as obtained
    from OPeNDAP. The full set of returned variables will include those
    requested by the end-user, and additional variables required to support
    those requested (e.g., grid dimension variables or CF-Convention
    metadata references).

    When the input Harmony message specifies a bounding box, shape file or
    named dimensions that require index-range subsetting, dimension
    variables will first be retrieved in a "prefetch" request to OPeNDAP.
    Then the bounding-box or shape file extents are converted to
    index-ranges. Similar behaviour occurs when a temporal range is
    requested by the end user, determining the indices of the temporal
    dimension from the prefetch response.

    Once the required variables, and index-ranges if needed, are derived,
    a request is made to OPeNDAP to retrieve only the requested data.

    """
    # Determine if index range subsetting will be required:
    request_is_index_subset = is_index_subset(harmony_message)

    # Produce map of variable dependencies with `earthdata-varinfo` and `.dmr`.
    varinfo = get_varinfo(
        opendap_url,
        output_dir,
        logger,
        harmony_source.shortName,
        harmony_message.accessToken,
        config,
    )

    # Obtain a list of all variables for the subset, including those used as
    # references by the requested variables.
    required_variables = get_required_variables(
        varinfo, harmony_source.variables, request_is_index_subset, logger
    )
    logger.info(
        'All required variables: ' f'{format_variable_set_string(required_variables)}'
    )

    # Define a cache to store all dimension index ranges (spatial, temporal):
    index_ranges = {}

    if request_is_index_subset:
        # Prefetch all dimension variables in full:
        dimensions_path = get_prefetch_variables(
            opendap_url,
            varinfo,
            required_variables,
            output_dir,
            logger,
            harmony_message.accessToken,
            config,
        )

        # Note regarding precedence of user requests ...
        # We handle the general dimension request first, in case the
        # user names a specific temporal or spatial dimension (e.g.,
        # "latitude").  If temporal/lat/lon args are also in the request, they
        # will override the index ranges derived from the requested dimension.
        if rgetattr(harmony_message, 'subset.dimensions', None) is not None:
            # Update `index_ranges` cache with ranges for the requested
            # dimension(s). This will convert the requested min and max
            # values to array indices in the proper order. Each item in
            # the dimension request is a list: [name, min, max]
            index_ranges.update(
                get_requested_index_ranges(
                    required_variables, varinfo, dimensions_path, harmony_message
                )
            )

        if (
            rgetattr(harmony_message, 'subset.bbox', None) is not None
            or rgetattr(harmony_message, 'subset.shape', None) is not None
        ):
            # Update `index_ranges` cache with ranges for horizontal grid
            # dimension variables (geographic and projected).
            shape_file_path = get_request_shape_file(
                harmony_message, output_dir, logger, config
            )

            index_ranges.update(
                get_spatial_index_ranges(
                    required_variables,
                    varinfo,
                    dimensions_path,
                    harmony_message,
                    shape_file_path,
                )
            )

        if harmony_message.temporal is not None:
            # Update `index_ranges` cache with ranges for temporal
            # variables. This will convert information from the temporal range
            # to array indices for each temporal dimension.
            index_ranges.update(
                get_temporal_index_ranges(
                    required_variables, varinfo, dimensions_path, harmony_message
                )
            )

    # Add any range indices to variable names for DAP4 constraint expression.
    variables_with_ranges = set(
        add_index_range(variable, varinfo, index_ranges)
        for variable in required_variables
    )
    logger.info(
        'variables_with_ranges: ' f'{format_variable_set_string(variables_with_ranges)}'
    )

    # Retrieve OPeNDAP data including only the specified variables in the
    # specified ranges.
    output_path = get_opendap_nc4(
        opendap_url,
        variables_with_ranges,
        output_dir,
        logger,
        harmony_message.accessToken,
        config,
    )

    # Fill the data outside the requested ranges for variables that cross a
    # dimensional discontinuity (for example longitude and the anti-meridian).
    fill_variables(output_path, varinfo, required_variables, index_ranges)

    return output_path


def get_varinfo(
    opendap_url: str,
    output_dir: str,
    logger: Logger,
    collection_short_name: str,
    access_token: str,
    config: Config,
) -> str:
    """Retrieve the `.dmr` from OPeNDAP and use `earthdata-varinfo` to
    populate a representation of the granule that maps dependencies between
    variables.

    """
    dmr_path = download_url(
        f'{opendap_url}.dmr.xml',
        output_dir,
        logger,
        access_token=access_token,
        config=config,
    )
    return VarInfoFromDmr(
        dmr_path, short_name=collection_short_name, config_file='hoss/hoss_config.json'
    )


def get_required_variables(
    varinfo: VarInfoFromDmr,
    variables: List[HarmonyVariable],
    request_is_index_subset: bool,
    logger: Logger,
) -> Set[str]:
    """Iterate through all requested variables from the Harmony message and
    extract their full paths. Then use the
    `VarInfoFromDmr.get_required_variables` method to also return all those
    variables that are required to support

    If index range subsetting is required, but no variables are specified
    (e.g., all variables are requested) then the requested variables should
    be set to all variables (science and non-science), so that index-range
    subsets can be specified in a DAP4 constraint expression.

    """
    requested_variables = set(
        (
            variable.fullPath
            if variable.fullPath.startswith('/')
            else f'/{variable.fullPath}'
        )
        for variable in variables
    )

    if request_is_index_subset and len(requested_variables) == 0:
        requested_variables = varinfo.get_science_variables().union(
            varinfo.get_metadata_variables()
        )

    logger.info(
        'Requested variables: ' f'{format_variable_set_string(requested_variables)}'
    )

    return varinfo.get_required_variables(requested_variables)


def fill_variables(
    output_path: str,
    varinfo: VarInfoFromDmr,
    required_variables: Set[str],
    index_ranges: IndexRanges,
) -> None:
    """Check the index ranges for all dimension variables. If the minimum
    index is greater than the maximum index in the subset range, then the
    requested dimension range crossed an edge of the grid (e.g. longitude),
    and must be filled in between those values.

    Note - longitude variables themselves will not be filled, to ensure
    valid grid coordinates at all points of the science variables.

    """
    fill_ranges = {
        dimension: index_range
        for dimension, index_range in index_ranges.items()
        if index_range[0] > index_range[1]
    }

    dimensions_to_fill = set(fill_ranges)

    if len(dimensions_to_fill) > 0:
        with Dataset(output_path, 'a', format='NETCDF4') as output_dataset:
            for variable_path in required_variables:
                fill_variable(
                    output_dataset,
                    fill_ranges,
                    varinfo,
                    variable_path,
                    dimensions_to_fill,
                )


def fill_variable(
    output_dataset: Dataset,
    fill_ranges: IndexRanges,
    varinfo: VarInfoFromDmr,
    variable_path: str,
    dimensions_to_fill: Set[str],
) -> None:
    """Check if the variable has dimensions that require filling. If so,
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
            get_fill_slice(dimension, fill_ranges) for dimension in variable.dimensions
        )

        output_dataset[variable_path][fill_index_tuple] = masked
