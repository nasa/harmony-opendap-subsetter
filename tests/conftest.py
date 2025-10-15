"""Set up fixtures for unit tests."""

import os
import tempfile
from logging import Logger
from unittest.mock import Mock

import h5py
import pytest
from varinfo import CFConfig, VarInfoFromNetCDF4


@pytest.fixture
def logger():
    """Logger fixture."""
    return Mock(spec=Logger)


@pytest.fixture
def mock_varinfo():
    """VarInfo fixture mock."""
    return Mock(spec=VarInfoFromNetCDF4)


@pytest.fixture
def smap_varinfo(sample_hdf5_file):
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
def sample_hdf5_file():
    """Create a temporary HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        temp_filename = tmp_file.name

    with h5py.File(temp_filename, 'w') as f:
        # Create basic HDF5 file with some test data
        f.create_dataset('dataset1', data=[1, 2, 3])
        f.create_dataset('dataset2', data=[4, 5, 6])
        f.attrs['attribute1'] = 'attribute_value1'
        f.attrs['attribute2'] = 'attribute_value2'

    yield temp_filename

    # Cleanup
    if os.path.exists(temp_filename):
        os.unlink(temp_filename)
