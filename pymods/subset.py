""" The module contains the main functions to perform variable subsetting on a
    single granule file. This should all be wrapped by the `subset_granule`
    function, which is called from the `HarmonyAdapter` class.

"""
from logging import Logger
from tempfile import mkdtemp

from harmony.message import Granule

from pymods.utilities import download_url
from pymods.var_info import VarInfoFromDmr, VarInfoFromPydap


VAR_INFO_SOURCE = 'dmr'


def subset_granule(granule: Granule, logger: Logger) -> str:
    """ This function takes a single Harmony Granule object, and extracts the
        requested variables, and those sub-variables they depend
        upon (such as coordinates), to produce an output file with only those
        variables. The path of this output file is returned.

    """
    granule_filename = granule.url.split('?')[0].rstrip('/').split('/')[-1]
    logger.info(f'Performing variable subsetting on: {granule_filename}')

    temp_dir = mkdtemp()

    # Create a list of requested variable full paths
    requested_variables = [variable.fullPath
                           if variable.fullPath.startswith('/')
                           else f'/{variable.fullPath}'
                           for variable in granule.variables]

    logger.info(f'Requested variables: {requested_variables}')

    # Harmony provides the OPeNDAP URL as the granule URL for this service
    # Determine whether to request the `.dmr` or to request the raw path from
    # `pydap`.
    if VAR_INFO_SOURCE == 'dmr':
        dmr_url = granule.url + '.dmr'
        dataset = VarInfoFromDmr(dmr_url, logger, temp_dir)
    else:
        dataset = VarInfoFromPydap(granule.url, logger, temp_dir)

    # Obtain a list of all variables for the subset, including those used as
    # references by the requested variables.
    required_variables = dataset.get_required_variables(set(requested_variables))
    logger.info(f'All required variables: {required_variables}')

    # TODO: Add switch mechanism for including (or not including) all metadata
    # variables in every subset request to OPeNDAP.

    # Make all required variable names compatible with an OPeNDAP subset URL.
    # Note: When using DAP4 (e.g. ".dap.nc4") we should be able to use the
    # original values for required_variables, from above, in the OPeNDAP URL.
    required_variables = [variable.lstrip('/').replace('/', '_')
                          for variable in required_variables]

    # TODO: Update URL to ".dap.nc4".
    opendap_url = f'{granule.url}.nc4?{",".join(required_variables)}'

    return download_url(opendap_url, temp_dir, logger, data='')
