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
    requested_variables = [f'/{variable.fullPath}' for variable in granule.variables]

    # Produce an output file that contains the variables identified in the
    # previous step.


    # Harmony provides the OPeNDAP URL as the granule URL for this service
    datasets = VarInfo(granule.url, logger)
    required_variables = datasets.get_required_variables(set(requested_variables))
    # TODO: Add switch mechanism for including (or not including) all metadata
    # variables in every subset request to OPeNDAP.

    # replace '/' with '_' in variable names
    required_variables = [variable.lstrip('/').replace('/', '_')
                          for variable in required_variables]

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
        out = open(output_file, 'wb')
        logger.info(f'Downloading output to {output_file}')
        out.write(result.content)
        out.close()
    except IOError:
        logger.error('Error occurred when downloading the file')
        raise IOError('Error occurred when downloading the file')

    return output_file
