""" The module contains the main functions to perform variable subsetting on a
    single granule file. This should all be wrapped by the `subset_granule`
    function, which is called from the `HarmonyAdapter` class.

"""
from logging import Logger

from harmony.message import Granule


def subset_granule(granule: Granule, logger: Logger) -> str:
    """ This function takes a single Harmony Granule object, and extracts the
        requested variables, and those sub-variables they depend
        upon (such as coordinates), to produce an output file with only those
        variables. The path of this output file is returned.

    """
    logger.info(f'Performing variable subsetting on: {granule.local_filename}')

    # Derive a list of required variables, both those in the Granule object and
    # their dependencies, such as coordinates.

    # Produce an output file that contains the variables identified in the
    # previous step.
    return '/path/to/subsetting/output.nc'
