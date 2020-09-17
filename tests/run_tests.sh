#!/bin/sh
####################################
#
# A script invoked by the test Dockerfile to run the Python `unittest` suite
# for the Harmonised variable subsetter. The script first runs the test suite,
# then it checks for linting errors, then checks for dependencies with known
# security vulnerabilities.
#
# Adapted from SwotRepr project, 2020-05-07
#
####################################

# Exit status used to report back to caller
STATUS=0

export HDF5_DISABLE_VERSION_CHECK=1

# Run the standard set of unit tests
coverage run -m unittest discover tests
RESULT=$?

if [ "$RESULT" -ne "0" ]; then
    STATUS=1
    echo "ERROR: unittest generated errors"
fi

echo "\n"
echo "Test Coverage Estimates"
coverage report --omit="*tests/*" 
coverage html --omit="*tests/*" -d /home/tests/coverage

# Run pylint
pylint pymods subsetter.py --disable=E0401
RESULT=$?
RESULT=$((3 & $RESULT))

if [ "$RESULT" -ne "0" ]; then
    STATUS=1
    echo "ERROR: pylint generated errors"
fi

# Run the python safety check
safety check
RESULT=$?

if [ "$RESULT" -ne "0" ]; then
    STATUS=1
    echo "ERROR: safety check generated errors"
fi

exit $STATUS
