""" The module contains the main functions to perform variable subsetting on a
    single granule file. This should all be wrapped by the `subset_granule`
    function, which is called from the `HarmonyAdapter` class.

"""
from logging import Logger
from typing import List

from harmony.message import Variable as HarmonyVariable
from harmony.util import Config
from varinfo import VarInfoFromDmr

from pymods.geo_grid import get_geo_bounding_box_subset
from pymods.utilities import download_url, get_opendap_nc4


def subset_granule(url: str, variables: List[HarmonyVariable], output_dir: str,
                   logger: Logger, access_token: str = None,
                   config: Config = None, bounding_box: List[float] = None) -> str:
    """ This function takes a granule's OPeNDAP URL and extracts the
        requested variables, and those sub-variables they depend
        upon (such as coordinates), to produce an output file with only those
        variables. The path of this output file is returned.

        The optional `bounding_box` argument can be supplied for geographically
        gridded data. In this case a bounding-box spatial subset will be
        applied to the retrieved variables, in addition to only retrieving the
        required variables.

    """
    granule_filename = url.split('?')[0].rstrip('/').split('/')[-1]
    logger.info(f'Performing variable subsetting on: {granule_filename}')

    # Create a list of requested variable full paths
    requested_variables = [variable.fullPath
                           if variable.fullPath.startswith('/')
                           else f'/{variable.fullPath}'
                           for variable in variables]

    logger.info(f'Requested variables: {requested_variables}')

    # Harmony provides the OPeNDAP URL as the granule URL for this service.
    # First download the `.dmr` representation of the file.
    dmr_path = download_url(f'{url}.dmr', output_dir, logger,
                            access_token=access_token, config=config)
    dataset = VarInfoFromDmr(dmr_path, logger,
                             config_file='pymods/var_subsetter_config.yml')

    # Obtain a list of all variables for the subset, including those used as
    # references by the requested variables.
    required_variables = dataset.get_required_variables(set(requested_variables))
    logger.info(f'All required variables: {required_variables}')

    # TODO: Add switch mechanism for including (or not including) all metadata
    # variables in every subset request to OPeNDAP.

    if bounding_box is not None:
        # Retrieve OPeNDAP data for a geographically gridded collection, that
        # includes only the specified variables, and only in the ranges defined
        # by the bounding box.
        output_path = get_geo_bounding_box_subset(required_variables, dataset,
                                                  bounding_box, url,
                                                  output_dir, logger,
                                                  access_token, config)
    else:
        # Retrieve OPeNDAP data including only the specified variables (but in
        # their full ranges).
        output_path = get_opendap_nc4(url, required_variables, output_dir,
                                      logger, access_token, config)

    return output_path
