from logging import Logger
from typing import Dict
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from pydap.model import BaseType, DatasetType
from webob.exc import HTTPClientError

from pymods.cf_config import CFConfig
from pymods.exceptions import PydapRetrievalError
from pymods.var_info import VarInfo, Variable


mock_variables = {
    '/ancillary_one': {'attributes': {'fullnamepath': '/ancillary_one'}},
    '/dimension_one': {'attributes': {'fullnamepath': '/dimensions_one'}},
    '/latitude': {'attributes': {'fullnamepath': '/latitude'},
                  'dimensions': ('/dimension_one', )},
    '/longitude': {'attributes': {'fullnamepath': '/longitude'},
                   'dimensions': ('/dimension_one', )},
    '/metadata_variable': {'attributes': {'fullnamepath': '/metadata_variable'}},
    '/science_variable': {
        'attributes': {'ancillary_variables': '/ancillary_one',
                       'coordinates': '/latitude, /longitude',
                       'fullnamepath': '/science_variable',
                       'subset_control_variables': '/subset_one'},
        'dimensions': ('/dimension_one', )
    },
    '/subset_one': {
        'attributes': {'coordinates': '/latitude, /longitude',
                       'fullnamepath': '/subset_one'},
        'dimensions': ('/dimension_one', )
    }
}

mock_fakesat_variables = {
    '/exclude_one/has_coordinates': {
        'attributes': {
            'fullnamepath': '/exclude_one/has_coordinates',
            'coordinates': '../science/latitude, ../science/longitude'
        }
    },
    '/required_group/has_no_coordinates': {
        'attributes': {'fullnamepath': '/required_group/has_no_coordinates'}
    },
    '/science/interesting_thing': {
        'attributes': {'fullnamepath': '/science/interesting_thing',
                       'coordinates': 'latitude, longitude'},
    },
    '/science/latitude': {'attributes': {'fullnamepath': '/science/latitude'}},
    '/science/longitude': {'attributes': {'fullnamepath': '/science/longitude'}},
}


def generate_pydap_response(variables: Dict[str, Dict[str, str]],
                            global_attributes: Dict) -> DatasetType:
    """ Create a pydap DatasetType with the requested variables and attributes,
        to mimic the output of a pydap.client.open_url request.

    """
    dataset = DatasetType()
    dataset.attributes = {'HDF5_GLOBAL': global_attributes}

    for variable_name, variable_properties in variables.items():
        variable_attributes = variable_properties.get('attributes', {})
        variable_dimensions = variable_attributes.get('dimensions', ())
        dataset[variable_name] = BaseType(variable_name, np.ones((2, 2)),
                                          attributes=variable_attributes,
                                          dimensions=variable_dimensions)

    return dataset


class TestVarInfo(TestCase):
    """ A class for testing the VarInfo class. """

    @classmethod
    def setUpClass(cls):
        """ Set up properties of the class that do not need to be reset between
            tests.

        """
        cls.logger = Logger('VarInfo tests')
        cls.pydap_url = 'http://test.opendap.org/opendap/hyrax/user/granule.h5'
        cls.config_file = 'tests/unit/data/test_config.yml'

        cls.mock_dataset = generate_pydap_response(mock_variables,
                                                   {'short_name': 'ATL03'})

        cls.mock_dataset_two = generate_pydap_response(mock_fakesat_variables,
                                                       {'short_name': 'FAKE99'})

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

        for global_attributes in test_attributes:
            with self.subTest(list(global_attributes.keys())[0]):
                mock_response = generate_pydap_response(
                    {'sea_surface_temp': {}}, global_attributes
                )
                mock_open_url.return_value = mock_response
                dataset = VarInfo(self.pydap_url, self.logger)

                mock_open_url.assert_called_once_with(self.pydap_url)
                self.assertEqual(dataset.short_name, short_name)

            mock_open_url.reset_mock()

        with self.subTest('No short name'):
            mock_response = generate_pydap_response({'sea_surface_temp': {}},
                                                    {})
            mock_open_url.return_value = mock_response
            dataset = VarInfo(self.pydap_url, self.logger)

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
                global_attributes = {'short_name': short_name}
                mock_response = generate_pydap_response(
                    {'sea_surface_temp': {}}, global_attributes
                )
                mock_open_url.return_value = mock_response
                dataset = VarInfo(self.pydap_url, self.logger)

                mock_open_url.assert_called_once_with(self.pydap_url)
                self.assertEqual(dataset.mission, expected_mission)

            mock_open_url.reset_mock()

    @patch('pymods.var_info.open_url')
    def test_var_info_instantiation_no_augmentation(self, mock_open_url):
        """ Ensure VarInfo instantiates correctly, creating records of all the
            variables in the granule, and correctly deciding if they are
            science variables, metadata or references. This test uses a mission
            and short name that do not have any CF overrides or supplements..

        """
        mock_open_url.return_value = self.mock_dataset
        dataset = VarInfo(self.pydap_url, self.logger,
                          config_file=self.config_file)

        self.assertEqual(dataset.short_name, 'ATL03')
        self.assertEqual(dataset.mission, 'ICESat2')
        self.assertEqual(dataset.global_attributes, {'short_name': 'ATL03'})

        self.assertEqual(set(dataset.metadata_variables.keys()),
                         {'/ancillary_one', '/dimensions_one', '/latitude',
                          '/longitude', '/metadata_variable'})
        self.assertEqual(set(dataset.variables_with_coordinates.keys()),
                         {'/science_variable', '/subset_one'})
        self.assertEqual(dataset.references, {'/ancillary_one', '/latitude',
                                              '/longitude', '/subset_one'})

    @patch('pymods.var_info.open_url')
    def test_var_info_instantiation_cf_augmentation(self, mock_open_url):
        """ Ensure VarInfo instantiates correcly, using a missions that has
            overrides and supplements in the CFConfig class.

        """

        mock_open_url.return_value = self.mock_dataset_two
        dataset = VarInfo(self.pydap_url, self.logger,
                          config_file=self.config_file)

        self.assertEqual(dataset.global_attributes,
                         {'short_name': 'FAKE99', 'global_override': 'GLOBAL',
                          'fakesat_global_supplement': 'fakesat value'})
        self.assertEqual(set(dataset.metadata_variables.keys()),
                         {'/science/latitude', '/science/longitude',
                          '/required_group/has_no_coordinates'})
        self.assertEqual(set(dataset.variables_with_coordinates.keys()),
                         {'/science/interesting_thing',
                          '/exclude_one/has_coordinates'})
        self.assertEqual(set(dataset.references), {'/science/latitude',
                                                   '/science/longitude'})

    @patch('pymods.var_info.open_url')
    def test_var_info_pydap_error(self, mock_open_url):
        """ Ensure VarInfo gracefully fails when a dataset object cannot be
            retrieved from pydap.

        """
        mock_open_url.side_effect = HTTPClientError('pydap problem')

        with self.assertRaises(PydapRetrievalError):
            VarInfo(self.pydap_url, self.logger)

    @patch('pymods.var_info.open_url')
    def test_var_info_get_science_variables(self, mock_open_url):
        """ Ensure the correct set of science variables is returned. This
            should account for excluded science variables defined in the
            associated instance of the CFConfig class.

        """
        mock_open_url.return_value = self.mock_dataset_two
        dataset = VarInfo(self.pydap_url, self.logger,
                          config_file=self.config_file)

        science_variables = dataset.get_science_variables()
        self.assertEqual(science_variables, {'/science/interesting_thing'})

    @patch('pymods.var_info.open_url')
    def test_var_info_get_metadata_variables(self, mock_open_url):
        """ Ensure the correct set of metadata variables (those without
            coordinate references) is returned. This should exclude variables
            that are also referred to by others via the metadata such as the
            coordinates attribute.

            This set should also include science variables that are explicitly
            excluded by the CFConfig instance.

        """
        mock_open_url.return_value = self.mock_dataset_two
        dataset = VarInfo(self.pydap_url, self.logger,
                          config_file=self.config_file)

        metadata_variables = dataset.get_metadata_variables()
        self.assertEqual(metadata_variables,
                         {'/required_group/has_no_coordinates',
                          '/exclude_one/has_coordinates'})

    @patch('pymods.var_info.open_url')
    def test_var_info_get_required_variables(self, mock_open_url):
        """ Ensure a full list of variables is returned when the VarInfo class
            is asked for those variables required to make a viable output
            granule. This should recursively search the references of all
            requested variables, to also include supporting variables such as
            coordinates, dimensions, ancillary_variables and
            subset_control_variables.

        """
        mock_open_url.return_value = self.mock_dataset_two
        dataset = VarInfo(self.pydap_url, self.logger,
                          config_file=self.config_file)

        required_variables = dataset.get_required_variables(
            {'/science/interesting_thing'}
        )
        self.assertEqual(required_variables, {'/required_group/has_no_coordinates',
                                              '/science/interesting_thing',
                                              '/science/latitude',
                                              '/science/longitude'})


class TestVariable(TestCase):
    """ Tests for the Variable class. """

    @classmethod
    def setUpClass(cls):
        """ Set up properties of the class that do not need to be reset between
            tests.

        """
        cls.config_file = 'tests/unit/data/test_config.yml'
        cls.fakesat_config = CFConfig('FakeSat', 'FAKE99',
                                      config_file=cls.config_file)
        cls.fakesat_map = {'dimensions': '/dimensions'}
        cls.variable_attributes = {
            'ancillary_variables': '/ancillary_data/epoch',
            'coordinates': 'latitude, longitude',
            'fullnamepath': '/group/variable',
            'subset_control_variables': 'begin count',
            'units': 'm'
        }
        cls.pydap_variable = BaseType('/group/variable', np.ones((2, 2)),
                                      attributes=cls.variable_attributes,
                                      dimensions=('dimensions',))

    def test_variable_instantiation(self):
        """ Ensure a Variable instance can be created from an input pydap
            BaseType instance.

        """
        variable = Variable(self.pydap_variable, self.fakesat_config,
                            self.fakesat_map)
        variable.full_name_path = '/group/variable'
        variable.group_path = '/group'
        variable.name = 'variable'
        self.assertEqual(variable.attributes.get('units'), 'm')
        self.assertEqual(variable.ancillary_variables, {'/ancillary_data/epoch'})
        self.assertEqual(variable.coordinates, {'/group/latitude',
                                                '/group/longitude'})
        self.assertEqual(variable.dimensions, {'/dimensions'})
        self.assertEqual(variable.subset_control_variables,
                         {'/group/begin', '/group/count'})

    def test_variable_cf_override(self):
        """ Ensure a CF attribute is overridden by the CFConfig value. """
        variable_attributes = {'coordinates': 'latitude, longitude',
                               'fullnamepath': '/coordinates_group/science'}
        pydap_variable = BaseType('/coordinates_group/science', np.ones((2, 2)),
                                  attributes=variable_attributes)
        pydap_map = {'/coordinates_group/science': '/coordinates_group/science'}

        variable = Variable(pydap_variable, self.fakesat_config, pydap_map)
        self.assertEqual(variable.coordinates, {'/coordinates_group/lat',
                                                '/coordinates_group/lon'})

    def test_variable_reference_qualification(self):
        """ Ensure different reference types (relative, absolute) are correctly
            qualified.

        """
        pydap_map = {'/gt1r/heights/bckgrd_mean': '/gt1r/heights/bckgrd_mean',
                     '/gt1r/latitude': '/gt1r/latitude',
                     '/gt1r/longitude': '/gt1r/longitude',
                     '/latitude': '/latitude',
                     '/longitude': '/longitude'}

        variable_name = '/gt1r/heights/bckgrd_mean'
        test_args = [['In parent group', '../latitude', '/gt1r/latitude'],
                     ['In granule root', '/latitude', '/latitude'],
                     ['Relative in same', './latitude', '/gt1r/heights/latitude'],
                     ['Basename only', 'latitude', '/gt1r/heights/latitude']]

        for description, coordinates, qualified_reference in test_args:
            with self.subTest(description):
                pydap_attributes = {'fullnamepath': variable_name,
                                    'coordinates': coordinates}
                pydap_variable = BaseType(variable_name, np.ones((2, 2)),
                                          attributes=pydap_attributes)
                variable = Variable(pydap_variable, self.fakesat_config,
                                    pydap_map)
                self.assertEqual(variable.coordinates, {qualified_reference})

    def test_variable_get_references(self):
        """ Ensure that a set of absolute paths to all variables referred to
            in the ancillary_variables, coordinates, dimensions and
            subset_control_variables is returned.

        """
        pydap_map = {'/science_variable': '/science_variable'}
        variable_name = '/science_variable'
        pydap_attributes = mock_variables[variable_name]['attributes']
        pydap_dimensions = mock_variables[variable_name]['dimensions']
        pydap_variable = BaseType(variable_name, np.ones((2, 2)),
                                  attributes=pydap_attributes,
                                  dimensions=pydap_dimensions)

        variable = Variable(pydap_variable, self.fakesat_config, pydap_map)
        references = variable.get_references()

        self.assertEqual(references, {'/ancillary_one', '/latitude',
                                      '/longitude', '/dimension_one',
                                      '/subset_one'})

    def test_pydap_dimension_conversion(self):
        """ Ensure that if a dimension has a pydap style name it is converted
            to the full path, for example:

            group_one_group_two_variable

            becomes:

            /group_one/group_two/variable

        """
        variable_name = 'group_one_variable'
        pydap_map = {'group_one_delta_time': '/group_one/delta_time',
                     'group_one_variable': '/group_one/variable'}
        pydap_attributes = {'fullnamepath': variable_name}
        pydap_variable = BaseType(variable_name, np.ones((2, 2)),
                                  attributes=pydap_attributes,
                                  dimensions=('group_one_delta_time',))
        variable = Variable(pydap_variable, self.fakesat_config, pydap_map)

        self.assertEqual(variable.dimensions, {'/group_one/delta_time'})
