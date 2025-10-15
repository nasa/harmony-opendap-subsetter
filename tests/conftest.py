"""Set up fixtures for unit tests."""

import os

import h5py
from logging import Logger
import pytest
import tempfile
from unittest.mock import Mock

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
    """VarInfo fixture for SMAP."""
    return VarInfoFromNetCDF4(
        sample_hdf5_file,
        config_file='hoss/hoss_config.json',
        short_name='SPL3FTA'
    )


@pytest.fixture(scope='function')
def sample_hdf5_file():
    """Create a temporary HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        temp_filename = tmp_file.name

    with h5py.File(temp_filename, 'w') as f:
        # Create basic HDF5 file with some test data
        f.create_dataset('dataset1', data=[1,2,3])
        f.create_dataset('dataset2', data=[4,5,6])
        f.attrs['attribute1'] = 'attribute_value1'
        f.attrs['attribute2'] = 'attribute_value2'

    yield temp_filename

    # Cleanup
    if os.path.exists(temp_filename):
        os.unlink(temp_filename)