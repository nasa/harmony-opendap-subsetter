from typing import Dict, List
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from pydap.model import BaseType, DatasetType

from pymods.var_info import VarInfo


def generate_pydap_response(variables: List[str],
                            attributes: Dict) -> DatasetType:
    """ Create a pydap DatasetType with the requested variables and attributes,
        to mimic the output of a pydap.client.open_url request.

    """
    dataset = DatasetType()
    dataset.attributes = {'HDF5_GLOBAL': attributes}

    for variable in variables:
        dataset[variable] = BaseType(variable, np.ones((2, 2)))

    return dataset


class TestVarInfo(TestCase):
    """ A class for testing the VarInfo class. """

    @classmethod
    def setUpClass(cls):
        """ Set up properties of the class that do not need to be reset between
            tests.

        """
        cls.pydap_url = 'http://test.opendap.org/opendap/hyrax/user/granule.h5'

    @patch('pymods.var_info.open_url')
    def test_var_info_short_name(self, mock_open_url):
        """ Ensure an instance of the VarInfo class is correctly initiated. """
        short_name = 'ATL03'

        test_attributes = [
            {'short_name': short_name},
            {'Metadata_DatasetIdentification': {'shortName': short_name}},
            {'METADATA_DatasetIdentification': {'shortName': short_name}},
            {'Metadata_SeriesIdentification': {'shortName': short_name}},
            {'METADATA_SeriesIdentification': {'shortName': short_name}},
        ]

        for attributes in test_attributes:
            with self.subTest(list(attributes.keys())[0]):
                mock_response = generate_pydap_response(['sea_surface_temp'],
                                                        attributes)
                mock_open_url.return_value = mock_response
                dataset = VarInfo(self.pydap_url)

                mock_open_url.assert_called_once_with(self.pydap_url)
                self.assertEqual(dataset.short_name, short_name)

            mock_open_url.reset_mock()

        with self.subTest('No short name'):
            mock_response = generate_pydap_response(['sea_surface_temp'], {})
            mock_open_url.return_value = mock_response
            dataset = VarInfo(self.pydap_url)

            mock_open_url.assert_called_once_with(self.pydap_url)
            self.assertEqual(dataset.short_name, None)
            mock_open_url.reset_mock()

    @patch('pymods.var_info.open_url')
    def test_var_info_mission(self, mock_open_url):
        """ Ensure VarInfo can identify the correct mission given a collection
            short name, or absence of one.

        """
        test_args = [['ATL03', 'ICESat2'],
                     ['GEDI_L1A', 'GEDI'],
                     ['GEDI01_A', 'GEDI'],
                     ['SPL3FTP', 'SMAP'],
                     ['VIIRS_NPP-OSPO-L2P-V2.3', 'VIIRS_PO'],
                     ['RANDOMSN', None],
                     [None, None]]

        for short_name, expected_mission in test_args:
            with self.subTest(short_name):
                attributes = {'short_name': short_name}
                mock_response = generate_pydap_response(['sea_surface_temp'],
                                                        attributes)
                mock_open_url.return_value = mock_response
                dataset = VarInfo(self.pydap_url)

                mock_open_url.assert_called_once_with(self.pydap_url)
                self.assertEqual(dataset.mission, expected_mission)

            mock_open_url.reset_mock()
