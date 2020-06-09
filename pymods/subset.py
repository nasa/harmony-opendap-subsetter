""" The module contains the main functions to perform variable subsetting on a
    single granule file. This should all be wrapped by the `subset_granule`
    function, which is called from the `HarmonyAdapter` class.

"""
from logging import Logger

from harmony.message import Granule
from pymods.utilities import cmr_query, get_token

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
    token = get_token()
    entry_title = cmr_query('collections', collection_id, 'EntryTitle', token)
    granule_ur = cmr_query('granules', granule_id, 'GranuleUR', token)

    # Derive a list of required variables, both those in the Granule object and
    # their dependencies, such as coordinates.

    # create a list of variable full paths
    variables = [variable.fullPath.replace('/', '_') for variable in granule.variables]

    # ToDo: call varInfo to get a list of required variables for subsetting

    # Produce an output file that contains the variables identified in the
    # previous step.

    # generate OPeNDAP URL
    opendap_url = f"https://opendap.uat.earthdata.nasa.gov/providers/{provider}/" \
                  f"collections/{entry_title}/granules/{granule_ur}.nc4?" \
                  f"{','.join(variables)}"

    print(f"{opendap_url}")

    return '/path/to/subsetting/output.nc'
