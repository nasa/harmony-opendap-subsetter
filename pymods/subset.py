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
    logger.info(f'Performing variable subsetting on: {granule.local_filename}')
    temp_dir = mkdtemp()
    file_root, file_ext = os.path.splitext(os.path.basename(granule.local_filename))
    output_file = temp_dir + os.sep + file_root + '_subset' + file_ext

    # Derive a list of required variables, both those in the Granule object and
    # their dependencies, such as coordinates.

    # create a list of variable full paths
    requested_variables = [f'/{variable.fullPath}' for variable in granule.variables]

    # Produce an output file that contains the variables identified in the
    # previous step.

    # generate OPeNDAP URL
    opendap_dmr_url = granule.url

    datasets = VarInfo(opendap_dmr_url)
    required_variables = datasets.get_required_variables(set(requested_variables))

    # replace '/' with '_' in variable names
    required_variables = [variable[1:].replace('/', '_') for variable in required_variables]

    opendap_url = f"{opendap_dmr_url}.nc4?{','.join(required_variables)}"

    try:
        result = requests.get(opendap_url)
        result.raise_for_status()
    except requests.HTTPError as err:
        logger.error(f'Request cannot be completed with error code {result.status_code}')
        raise requests.HTTPError(f'Request cannot be completed with error code {result.status_code}')

    try:
        out = open(output_file, 'wb')
        logger.info(f'Downloading output to {output_file}')
        out.write(result.content)
        out.close()
    except IOError:
        logger.error('Error occurred when downloading the file')
        raise IOError('Error occurred when downloading the file')

    return output_file
