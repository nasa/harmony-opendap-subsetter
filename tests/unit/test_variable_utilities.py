"""This module contains unit tests for the variable_utilities.py

"""

import pytest

from harmony_service_lib.message import Variable as HarmonyVariable

from hoss.exceptions import OnlyInvalidVariablesRequested
from hoss.variable_utilities import (
    get_processable_variables,
    get_excluded_variables,
    is_excluded_science_variable
)


def test_get_processable_variables_contains_string_variables(mocker, mock_varinfo, logger):
    """This test checks that requested string variables that exist in the
    exclusions listed in the varinfo configuration file are not returned.

    """
    requested_variable_paths = {
        'string_variable_time_utc',
        'group1/nested_string_variable_time_utc',
        'non_string_variable'
    }

    requested_harmony_variables = [
        HarmonyVariable({'fullPath': variable_path}) for variable_path in requested_variable_paths
    ]

    required_variables = {
        '/string_variable_time_utc',
        '/group1/nested_string_variable_time_utc',
        '/non_string_variable',
        '/coordinate1'  # Additional CF variable
    }

    mock_get_excluded_variables = mocker.patch(
        'hoss.variable_utilities.get_excluded_variables',
        return_value=set([
            '/string_variable_time_utc',
            '/group1/nested_string_variable_time_utc'
        ])
    )

    expected_output = {'/non_string_variable', '/coordinate1'}

    actual_output = get_processable_variables(
        required_variables,
        requested_harmony_variables,
        mock_varinfo,
        logger
    )

    mock_get_excluded_variables.assert_called_once()
    assert expected_output == actual_output


def test_get_processable_variables_contains_no_string_variables(mocker, mock_varinfo, logger):
    """This test checks that the output string set matches the input string set
    when no string variables are included in the request.

    """
    requested_variable_paths = {
        'non_string_variable'
    }

    requested_harmony_variables = [
        HarmonyVariable({'fullPath': variable_path}) for variable_path in requested_variable_paths
    ]

    required_variables = {
        '/non_string_variable',
        '/coordinate1'  # Additional CF variable
    }

    mock_get_excluded_variables = mocker.patch(
        'hoss.variable_utilities.get_excluded_variables',
        return_value=set()
    )

    expected_output = {'/non_string_variable', '/coordinate1'}

    actual_output = get_processable_variables(
        required_variables,
        requested_harmony_variables,
        mock_varinfo,
        logger
    )

    mock_get_excluded_variables.assert_called_once()
    assert expected_output == actual_output


def test_get_processable_variables_exception(mocker, mock_varinfo, logger):
    """This test checks that the OnlyInvalidVariablesRequested exception is thrown
    when only string variables are requested.

    """
    requested_variable_paths = {
        'string_variable_time_utc',
        'group1/nested_string_variable_time_utc',
    }

    requested_harmony_variables = [
        HarmonyVariable({'fullPath': variable_path}) for variable_path in requested_variable_paths
    ]

    required_variables = {
        '/string_variable_time_utc',
        '/group1/nested_string_variable_time_utc',
        '/coordinate1'  # Additional CF variable
    }

    mock_get_excluded_variables = mocker.patch(
        'hoss.variable_utilities.get_excluded_variables',
        return_value=set([
            '/string_variable_time_utc',
            '/group1/nested_string_variable_time_utc'
        ])
    )

    with pytest.raises(OnlyInvalidVariablesRequested):
        get_processable_variables(
            required_variables,
            requested_harmony_variables,
            mock_varinfo,
            logger
        )

    mock_get_excluded_variables.assert_called_once()


def test_get_excluded_variables(smap_varinfo):
    """This test checks that only variables listed in the varinfo configuration's
    ExcludedScienceVariables section are excluded.

    """
    variables = {
        '/string_time_utc_seconds',
        '/group/nested_time_utc_string',
        '/unexcluded_string_time',
        '/Freeze_Thaw_Retrieval_Data/freeze_reference_date',
        '/string_variable',
        '/group/nested_string_variable'
    }

    expected_output = {
        '/string_time_utc_seconds',
        '/group/nested_time_utc_string',
        '/Freeze_Thaw_Retrieval_Data/freeze_reference_date',
    }

    actual_output = get_excluded_variables(smap_varinfo, variables)

    assert expected_output == actual_output


def test_is_excluded_science_variable(smap_varinfo):
    """This test checks that the excluded science variables listed in
    configuration file are excluded when requested.

    """
    assert is_excluded_science_variable(
        smap_varinfo, '/string_time_utc_seconds'
    )
    assert is_excluded_science_variable(
        smap_varinfo, '/group/nested_time_utc_string'
    )
    assert is_excluded_science_variable(
        smap_varinfo, '/Freeze_Thaw_Retrieval_Data/freeze_reference_date'
    )
    assert not is_excluded_science_variable(
        smap_varinfo, '/string_variable'
    )
    assert not is_excluded_science_variable(
        smap_varinfo, '/group/nested_string_variable'
    )
