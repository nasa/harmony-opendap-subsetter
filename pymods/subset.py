""" The module contains the main functions to perform variable subsetting on a
    single granule file. This should all be wrapped by the `subset_granule`
    function, which is called from the `HarmonyAdapter` class.

"""
import os
from logging import Logger
from tempfile import mkdtemp
import requests

from harmony.message import Granule
from pymods.var_info import VarInfo


VAR_INFO_SOURCE = 'dmr'


def subset_granule(granule: Granule, logger: Logger) -> str:
    """ This function takes a single Harmony Granule object, and extracts the
        requested variables, and those sub-variables they depend
        upon (such as coordinates), to produce an output file with only those
        variables. The path of this output file is returned.

    """
    granule_filename = granule.url.split('?')[0].rstrip('/').split('/')[-1]
    logger.info(f'Performing variable subsetting on: {granule_filename}')

    granule_basename = os.path.splitext(granule_filename)[0]
    file_ext = '.nc4'
    temp_dir = mkdtemp()
    output_file = os.sep.join([temp_dir, granule_basename + '_subset' + file_ext])

    # Derive a list of required variables, both those in the Granule object and
    # their dependencies, such as coordinates.

    # create a list of variable full paths
    requested_variables = [variable.fullPath
                           if variable.fullPath.startswith('/')
                           else f'/{variable.fullPath}'
                           for variable in granule.variables]

    logger.info(f'Requested variables: {requested_variables}')

    # Harmony provides the OPeNDAP URL as the granule URL for this service
    if VAR_INFO_SOURCE == 'dmr':
        var_info_url = granule.url + '.dmr'
    else:
        var_info_url = granule.url

    datasets = VarInfo(var_info_url, logger)
    required_variables = datasets.get_required_variables(set(requested_variables))
    logger.info(f'All required variables: {required_variables}')

    # TODO: Add switch mechanism for including (or not including) all metadata
    # variables in every subset request to OPeNDAP.

    # replace '/' with '_' in variable names
    required_variables = [variable.lstrip('/').replace('/', '_')
                          for variable in required_variables]

    # TODO: Update URL to ".dap.nc4".
    opendap_url = f"{granule.url}.nc4?{','.join(required_variables)}"

    try:
        result = requests.get(opendap_url)
        result.raise_for_status()
    except requests.HTTPError:
        logger.error('Request cannot be completed with error code '
                     f'{result.status_code}')
        raise requests.HTTPError('Request cannot be completed with error code '
                                 f'{result.status_code}')

    try:
        logger.info(f'Downloading output to {output_file}')
        with open(output_file, 'wb') as file_handler:
            file_handler.write(result.content)
    except IOError:
        logger.error('Error occurred when downloading the file')
        raise IOError('Error occurred when downloading the file')

    return output_file
