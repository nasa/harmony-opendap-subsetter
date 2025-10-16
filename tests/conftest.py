"""Set up fixtures for unit tests."""

import logging
from unittest.mock import Mock

import pytest
from netCDF4 import Dataset
from varinfo import VarInfoFromNetCDF4


@pytest.fixture(scope='function')
def logger():
    """Logger fixture."""
    return logging.getLogger("test_logger")


@pytest.fixture(scope='function')
def mock_varinfo():
    """VarInfo fixture mock."""
    return Mock(spec=VarInfoFromNetCDF4)


@pytest.fixture(scope='function')
def SPL3FTA_varinfo(sample_hdf5_file):
    """VarInfo fixture for SMAP.

    The SPL3FTA collection is used to create the varinfo object because it's
    the only SMAP L3 collection that contains excluded string variables
    that don't match the string variable pattern of all the other SMAP L3
    strings. This way, when checking for the exclusions applied to SMAP,
    all existing SMAP exclusion variable patterns will be checked.

    """
    return VarInfoFromNetCDF4(
        sample_hdf5_file, config_file='hoss/hoss_config.json', short_name='SPL3FTA'
    )


@pytest.fixture(scope='function')
def sample_hdf5_file(tmp_path):
    """Create a temporary HDF5 file for testing.

    This sample file is empty since no tests currently access data in the
    sample file.

    """
    temp_filename = tmp_path / "test.h5"

    # Create an empty HDF5 file.
    with Dataset(temp_filename, 'w'):
        pass

    return str(temp_filename)
