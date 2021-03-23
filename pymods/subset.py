""" The module contains the main functions to perform variable subsetting on a
    single granule file. This should all be wrapped by the `subset_granule`
    function, which is called from the `HarmonyAdapter` class.

"""
from logging import Logger
from typing import List
from urllib.parse import urlencode

from harmony.message import Variable
from harmony.util import Config
from varinfo import VarInfo

from pymods.utilities import download_url


def subset_granule(
        url: str,
        variables: List[Variable],
        output_dir: str,
        logger: Logger,
        access_token: str = None,
        config: Config = None) -> str:
    """ This function takes a granule's OPeNDAP URL and extracts the
        requested variables, and those sub-variables they depend
        upon (such as coordinates), to produce an output file with only those
        variables. The path of this output file is returned.

    """
    granule_filename = url.split('?')[0].rstrip('/').split('/')[-1]
    logger.info(f'Performing variable subsetting on: {granule_filename}')

    # Create a list of requested variable full paths
    requested_variables = [variable.fullPath
                           if variable.fullPath.startswith('/')
                           else f'/{variable.fullPath}'
                           for variable in variables]

    logger.info(f'Requested variables: {requested_variables}')

    # Harmony provides the OPeNDAP URL as the granule URL for this service
    dmr_url = url + '.dmr'
    dataset = VarInfo(dmr_url, logger, output_dir, access_token, config,
                      config_file='pymods/var_subsetter_config.yml')

    # Obtain a list of all variables for the subset, including those used as
    # references by the requested variables.
    required_variables = dataset.get_required_variables(set(requested_variables))
    logger.info(f'All required variables: {required_variables}')

    # TODO: Add switch mechanism for including (or not including) all metadata
    # variables in every subset request to OPeNDAP.

    # Build the DAP4 format constraint expression, which is a semi-colon
    # separated list of variable names.
    # This should be in the request body, not a query string parameter.
    constraint_expression = urlencode({'dap4.ce': ';'.join(required_variables)})
    opendap_url = f'{url}.dap.nc4?{constraint_expression}'

    return download_url(opendap_url, output_dir, logger,
                        access_token=access_token, config=config)
