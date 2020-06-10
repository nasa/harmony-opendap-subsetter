""" The module contains the main functions to perform variable subsetting on a
    single granule file. This should all be wrapped by the `subset_granule`
    function, which is called from the `HarmonyAdapter` class.

"""
from logging import Logger

from harmony.message import Granule
from pymods.utilities import cmr_query, get_token
from pymods.var_info import VarInfo

def subset_granule(granule: Granule, logger: Logger) -> str:
    """ This function takes a single Harmony Granule object, and extracts the
        requested variables, and those sub-variables they depend
        upon (such as coordinates), to produce an output file with only those
        variables. The path of this output file is returned.

    """
    logger.info(f'Performing variable subsetting on: {granule.local_filename}')
    collection_id = granule.collection
    granule_id = granule.id

    # abstract provider from collection concept id
    provider = collection_id.partition('-')[2]

    # get collection EntryTitle and granule UR from CMR query
    token = get_token(logger)
    if not token:
        raise Exception("Failed to generate CMR token")

    entry_title = cmr_query('collections', collection_id, 'EntryTitle', token, logger)

    if not entry_title:
        raise Exception("Failed to obtain EntryTitle from CMR")

    granule_ur = cmr_query('granules', granule_id, 'GranuleUR', token, logger)

    if not granule_ur:
        raise Exception("Failed to obtain granule UR from CMR")


    # Derive a list of required variables, both those in the Granule object and
    # their dependencies, such as coordinates.

    # create a list of variable full paths
    requested_variables = [f'/{variable.fullPath}' for variable in granule.variables]

    # Produce an output file that contains the variables identified in the
    # previous step.

    # generate OPeNDAP URL
    opendap_dmr_url = f"https://opendap.uat.earthdata.nasa.gov/providers/{provider}/" \
                      f"collections/{entry_title}/granules/{granule_ur}.h5"

    datasets = VarInfo(opendap_dmr_url)
    required_variables = datasets.get_required_variables(set(requested_variables))

    # replace '/' with '_' in variable names
    required_variables = [variable[1:].replace('/', '_') for variable in required_variables]

    opendap_url = f"{opendap_dmr_url}?{','.join(required_variables)}"

    return '/path/to/subsetting/output.nc'
