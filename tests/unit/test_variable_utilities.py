"""This module contains unit tests for the variable_utilities.py"""

import logging

import pytest
from harmony_service_lib.message import Variable as HarmonyVariable

from hoss.exceptions import InvalidVariableRequest
from hoss.variable_utilities import (
    check_invalid_variable_request,
    get_excluded_variables,
    is_excluded_science_variable,
)


def test_check_invalid_variable_request_exclusions(mocker, mock_varinfo, logger):
    """This test checks that an exception is thrown when an excluded science
    variable in the varinfo config file is explicitly requested.

    """
    excluded_string1 = 'string_variable_time_utc'
    excluded_string2 = 'subgroup/nested_string_variable_time_utc'
    requested_variable_paths = {
        excluded_string1,
        excluded_string2,
        'non_string_variable',
    }

    requested_harmony_variables = [
        HarmonyVariable({'fullPath': variable_path})
        for variable_path in requested_variable_paths
    ]

    mock_get_excluded_variables = mocker.patch(
        'hoss.variable_utilities.get_excluded_variables',
        return_value=set([excluded_string1, excluded_string2]),
    )

    with pytest.raises(InvalidVariableRequest) as excinfo:
        check_invalid_variable_request(
            requested_harmony_variables, mock_varinfo, logger
        )

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

    excluded_vars = {'excluded_var1', 'excluded_var2'}
    mock_get_excluded_variables = mocker.patch(
        'hoss.variable_utilities.get_excluded_variables',
        return_value=excluded_vars,
    )

    # Set caplog to capture INFO level logs.
    with caplog.at_level(logging.INFO):
        check_invalid_variable_request(
            requested_harmony_variables, mock_varinfo, logger
        )

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

    """
    logger = logging.getLogger("test_logger")
    requested_variable_paths = {
        'non_string_variable',
        'subgroup/nested_non_string_variable',
    }

    requested_harmony_variables = [
        HarmonyVariable({'fullPath': variable_path})
        for variable_path in requested_variable_paths
    ]

    excluded_vars = {'excluded_var1', 'excluded_var2'}
    mock_get_excluded_variables = mocker.patch(
        'hoss.variable_utilities.get_excluded_variables',
        return_value=excluded_vars,
    )

    # Set caplog to capture INFO level logs.
    with caplog.at_level(logging.INFO):
        check_invalid_variable_request(
            requested_harmony_variables, mock_varinfo, logger
        )

    # Check the log message
    assert 'No invalid variables are requested.' in caplog.text

    mock_get_excluded_variables.assert_called_once()


def test_get_excluded_variables(SPL3FTA_varinfo):
    """This test checks that only variables listed in the varinfo configuration's
    ExcludedScienceVariables section are excluded.

    """
    variables = {
        '/string_time_utc_seconds',
        '/group/nested_time_utc_string',
        '/unexcluded_string_time',
        '/Freeze_Thaw_Retrieval_Data/freeze_reference_date',
        '/string_variable',
        '/group/nested_string_variable',
    }

    expected_output = {
        '/string_time_utc_seconds',
        '/group/nested_time_utc_string',
        '/Freeze_Thaw_Retrieval_Data/freeze_reference_date',
    }

    actual_output = get_excluded_variables(SPL3FTA_varinfo, variables)

    assert expected_output == actual_output


def test_is_excluded_science_variable(SPL3FTA_varinfo):
    """This test checks that the excluded science variables listed in
    configuration file are excluded when requested.

    """
    assert is_excluded_science_variable(SPL3FTA_varinfo, '/string_time_utc_seconds')
    assert is_excluded_science_variable(
        SPL3FTA_varinfo, '/group/nested_time_utc_string'
    )
    assert is_excluded_science_variable(
        SPL3FTA_varinfo, '/Freeze_Thaw_Retrieval_Data/freeze_reference_date'
    )
    assert not is_excluded_science_variable(SPL3FTA_varinfo, '/string_variable')
    assert not is_excluded_science_variable(
        SPL3FTA_varinfo, '/group/nested_string_variable'
    )
