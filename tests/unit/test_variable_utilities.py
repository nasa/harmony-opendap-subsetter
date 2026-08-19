"""This module contains unit tests for the variable_utilities.py"""

import logging

import pytest
from harmony_service_lib.message import Variable as HarmonyVariable
from varinfo import VarInfoFromDmr

from hoss.exceptions import InvalidVariableRequest
from hoss.variable_utilities import (
    check_invalid_variable_request,
)


def test_check_invalid_variable_request_exclusions(mocker, mock_varinfo):
    """This test checks that an exception is thrown when an excluded science
    variable in the varinfo config file is explicitly requested.

    It also checks that variable names are compared regardless of
    whether or not they include leading slashes.

    """
    excluded_string1 = '/excluded_string_variable'
    excluded_string2 = 'subgroup/nested_excluded_variable'
    requested_variable_paths = {
        excluded_string1,
        excluded_string2,
        'non_string_variable',
    }

    requested_harmony_variables = [
        HarmonyVariable({'fullPath': variable_path})
        for variable_path in requested_variable_paths
    ]

    # Returned excluded variables should always have leading slashes.
    excluded_vars = {
        '/excluded_string_variable',
        '/subgroup/nested_excluded_variable',
    }
    mock_get_excluded_variables = mocker.patch.object(
        mock_varinfo,
        'get_excluded_science_variables',
        return_value=excluded_vars,
    )

    with pytest.raises(InvalidVariableRequest) as excinfo:
        check_invalid_variable_request(requested_harmony_variables, mock_varinfo)

    # Check that the excluded variables are in the exception message.
    # Since it's an unordered set converted to string, check individually.
    error_msg = str(excinfo.value)
    assert excluded_string1 in error_msg
    assert excluded_string2 in error_msg
    assert (
        "Some variables requested are not supported and could not be processed:"
        in error_msg
    )

    mock_get_excluded_variables.assert_called_once()


def test_check_invalid_variable_request_all(mocker, mock_varinfo, logger, caplog):
    """This test checks that no exception is thrown when there is not an
    explicit variable request by checking the expected logger message.

    """
    requested_harmony_variables = set()  # Empty set triggers "all variables" path

    excluded_vars = {'/excluded_var1', '/excluded_var2'}
    mock_get_excluded_variables = mocker.patch.object(
        mock_varinfo,
        'get_excluded_science_variables',
        return_value=excluded_vars,
    )

    # Set caplog to capture INFO level logs.
    with caplog.at_level(logging.INFO):
        check_invalid_variable_request(requested_harmony_variables, mock_varinfo)

    # Check the log message
    assert (
        'All variables are requested. The following variables will be excluded:'
        in caplog.text
    )

    # Check that the excluded variables appear in the log.
    # Since it's an unordered set converted to string, check individually.
    assert 'excluded_var1' in caplog.text
    assert 'excluded_var2' in caplog.text

    mock_get_excluded_variables.assert_called_once()


def test_check_invalid_variable_request_no_exclusions(
    mocker, mock_varinfo, logger, caplog
):
    """This test checks that no exception is thrown when no excluded variables
    are requested by checking the expected logger message.

    It also checks that variable names are compared regardless of
    whether or not they include leading slashes.

    """
    requested_variable_paths = {
        '/non_string_variable',
        'subgroup/nested_non_string_variable',
    }

    requested_harmony_variables = [
        HarmonyVariable({'fullPath': variable_path})
        for variable_path in requested_variable_paths
    ]

    excluded_vars = {'/excluded_var1', '/excluded_var2'}
    mock_get_excluded_variables = mocker.patch.object(
        mock_varinfo,
        'get_excluded_science_variables',
        return_value=excluded_vars,
    )

    # Set caplog to capture INFO level logs.
    with caplog.at_level(logging.INFO):
        check_invalid_variable_request(requested_harmony_variables, mock_varinfo)

    # Check the log message
    assert 'No invalid variables are requested.' in caplog.text

    mock_get_excluded_variables.assert_called_once()


def test_smap_l3_time_utc_variables_not_excluded():
    """UTC time string variables must no longer be excluded for SMAP L3
    collections.

    """
    varinfo = VarInfoFromDmr(
        'tests/data/SC_SPL3SMP_008.dmr',
        'SPL3SMP',
        config_file='hoss/hoss_config.json',
    )

    excluded_variables = varinfo.get_excluded_science_variables()
    time_utc_excluded = {
        variable for variable in excluded_variables if 'time_utc' in variable
    }
    assert time_utc_excluded == set()

    # An explicit request for a UTC time string variable must not raise.
    requested_harmony_variables = [
        HarmonyVariable({'fullPath': '/Soil_Moisture_Retrieval_Data_AM/tb_time_utc'}),
        HarmonyVariable(
            {'fullPath': '/Soil_Moisture_Retrieval_Data_PM/tb_time_utc_pm'}
        ),
    ]
    check_invalid_variable_request(requested_harmony_variables, varinfo)
