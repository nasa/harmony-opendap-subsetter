"""Variable helper functions.

A collection of functions for getting information about variables, mostly with
varinfo.

"""

from typing import Set

from harmony_service_lib.message import Variable as HarmonyVariable
from varinfo import VarInfoFromNetCDF4

from hoss.exceptions import InvalidVariableRequest
from hoss.harmony_log_context import get_logger
from hoss.utilities import format_variable_set_string


def check_invalid_variable_request(
    requested_variables: Set[HarmonyVariable],
    varinfo: VarInfoFromNetCDF4,
) -> None:
    """Check if explicitly requested variables are listed as excluded in
    the varinfo configuration, and if so throw an exception listing the
    invalid requested variables.

    """
    requested_variable_paths = {
        f'/{v.fullPath.lstrip("/")}' for v in requested_variables
    }
    unprocessable_variables = varinfo.get_excluded_science_variables()

    # If no variables are requested, all variables will be returned and the
    # varinfo exclusions will automatically be applied.
    if not requested_variables:
        get_logger().info(
            f'All variables are requested. The following variables will be excluded:'
            f'{unprocessable_variables}'
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

    get_logger().info('No invalid variables are requested.')
    return
