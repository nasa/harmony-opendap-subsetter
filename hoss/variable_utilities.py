"""Variable helper functions.

A collection of functions for getting information about variables, mostly with
varinfo.

"""

import re
from logging import Logger
from typing import Set

from harmony_service_lib.message import Variable as HarmonyVariable
from varinfo import VarInfoFromNetCDF4

from hoss.exceptions import InvalidVariableRequest
from hoss.utilities import format_variable_set_string


def check_invalid_variable_request(
    requested_variables: Set[HarmonyVariable],
    varinfo: VarInfoFromNetCDF4,
    logger: Logger,
) -> None:
    """Check if explicitly requested variables are listed as excluded in
    the varinfo configuration, and if so throw an exception listing the
    invalid requested variables.

    """
    # A leading slash must be added to the requested variable paths since the
    # excluded variables have leading slashes.
    requested_variable_paths = {f'/{v.fullPath}' for v in requested_variables}
    unprocessable_variables = get_excluded_variables(varinfo, requested_variable_paths)

    # If no variables are requested, all variables will be returned and the
    # varinfo exclusions will automatically be applied.
    if not requested_variables:
        logger.info(
            f'All variables are requested. The following variables will be excluded: {unprocessable_variables}'
        )
        return

    # Check if any of the requested variables are unprocessable.
    # If so, throw an error.
    requested_unprocessable_variables = unprocessable_variables.intersection(
        requested_variable_paths
    )

    if requested_unprocessable_variables:
        raise InvalidVariableRequest(
            format_variable_set_string(requested_unprocessable_variables)
        )

    logger.info('No invalid variables are requested.')
    return


def get_excluded_variables(
    var_info: VarInfoFromNetCDF4, variables_list: Set[str]
) -> Set[str]:
    """Input variables that can't be processed by HOSS.

    This includes the excluded science variables specified in the varinfo
    configuration file.

    """
    excluded_vars = {
        var for var in variables_list if is_excluded_science_variable(var_info, var)
    }

    return excluded_vars


def is_excluded_science_variable(
    var_info: VarInfoFromNetCDF4, variable_name: str
) -> bool:
    """Returns True if variable is explicitly excluded by varinfo configuration."""
    exclusions_pattern = re.compile(
        '|'.join(var_info.cf_config.excluded_science_variables)
    )
    return var_info.variable_is_excluded(variable_name, exclusions_pattern)
