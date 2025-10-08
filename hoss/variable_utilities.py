"""Variable helper functions.

A collection of functions for getting information about variables, mostly with
varinfo.

"""

from logging import Logger
import re
from typing import Set

from harmony_service_lib.message import Variable as HarmonyVariable
from varinfo import VarInfoFromNetCDF4

from hoss.exceptions import OnlyInvalidVariableRequest
from hoss.utilities import format_variable_set_string


def get_processable_variables(
    required_variables: Set[str],
    requested_variables: Set[HarmonyVariable],
    varinfo: VarInfoFromNetCDF4,
    logger: Logger
) -> Set[str]:
    """Return only variables that HOSS can process.

    This removes variables listed as excluded science variables in the varinfo
    configuration file.

    """
    requested_variable_paths = {f'/{v.fullPath}' for v in requested_variables}
    unprocessable_variables = get_unprocessable_variables(varinfo, requested_variable_paths)

    # Throw an error when the request contains only unprocessable variables.
    if (requested_variable_paths
            and len(unprocessable_variables) == len(requested_variable_paths)):
        raise OnlyInvalidVariableRequest(format_variable_set_string(requested_variable_paths))

    # Remove unprocessable variables.
    if unprocessable_variables.intersection(requested_variable_paths):
        logger.info(f'Dropping unprocessable variables: {unprocessable_variables}')
        required_variables -= unprocessable_variables

    return required_variables


def get_unprocessable_variables(
    var_info: VarInfoFromNetCDF4, var_list: Set[str]
) -> Set[str]:
    """Input variables that can't be processed by HOSS.

    This includes the excluded science variables specified in the varinfo
    configuration file.

    """
    excluded_vars = {
        var for var in var_list if is_excluded_science_variable(var_info, var)
    }

    return excluded_vars


def is_excluded_science_variable(var_info: VarInfoFromNetCDF4, var) -> bool:
    """Returns True if variable is explicitly excluded by varinfo configuration."""
    exclusions_pattern = re.compile(
        '|'.join(var_info.cf_config.excluded_science_variables)
    )
    return var_info.variable_is_excluded(var, exclusions_pattern)
