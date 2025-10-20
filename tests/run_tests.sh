#!/bin/bash
####################################
#
# A script invoked by the test Dockerfile to run the Python `unittest` suite
# for the Harmony OPeNDAP SubSetter (HOSS). The script first runs the test
# suite, then it checks for linting errors.
#
# 2020-05-07: Adapted from SwotRepr project.
# 2022-01-03: Removed safety checks, as these are now run in Snyk.
# 2023-10-06: Updated pylint directory scanned to "hoss".
#
####################################

# Exit status used to report back to caller
STATUS=0

export HDF5_DISABLE_VERSION_CHECK=1

echo -e "\nRunning tests..."

pytest ./tests -s --cov=hoss --junitxml=test-reports/pytest-results.xml --cov-report=html:coverage --cov-report term

RESULT=$?

if [ "$RESULT" -ne "0" ]; then
    STATUS=1
    echo "ERROR: tests generated errors"
fi

echo -e "\n"

# Run pylint
# Ignored errors/warnings:
# W1203 - use of f-strings in log statements. This warning is leftover from
#         using ''.format() vs % notation. For more information, see:
#     	  https://github.com/PyCQA/pylint/issues/2354#issuecomment-414526879
pylint hoss --disable=W1203 --extension-pkg-whitelist=netCDF4
RESULT=$?
RESULT=$((3 & $RESULT))

if [ "$RESULT" -ne "0" ]; then
    STATUS=1
    echo "ERROR: pylint generated errors"
fi

exit $STATUS
