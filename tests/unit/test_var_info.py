from logging import Logger
from typing import Dict
from unittest import TestCase
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from pydap.model import BaseType, DatasetType
from requests import HTTPError
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
    dataset.attributes = global_attributes

    for variable_name, variable_properties in variables.items():
        variable_attributes = variable_properties.get('attributes', {})
        variable_dimensions = variable_attributes.get('dimensions', ())
        dataset[variable_name] = BaseType(variable_name, np.ones((2, 2)),
                                          attributes=variable_attributes,
                                          dimensions=variable_dimensions)

    return dataset


class MockResponse:
    """ A test class to be used in mocking a response from the `requests.get`
        method.

    """
    def __init__(self, status_code: int, content: str):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        """ Check the response status code. If it isn't in the range of
            expected successful status codes, then raise an exception.

        """
        if self.status_code > 299 or self.status_code < 200:
            raise HTTPError('Could not retrieve data.')


class TestVarInfoPydap(TestCase):
    """ A class for testing the VarInfo class with a `pydap` URL. """

    @classmethod
    def setUpClass(cls):
        """ Set up properties of the class that do not need to be reset between
            tests.

        """
        cls.logger = Logger('VarInfo tests')
        cls.pydap_url = 'http://test.opendap.org/opendap/hyrax/user/granule.h5'
        cls.config_file = 'tests/unit/data/test_config.yml'

        cls.mock_dataset = generate_pydap_response(
            mock_variables,
            {'HDF5_GLOBAL': {'short_name': 'ATL03'}}
        )

        cls.mock_dataset_two = generate_pydap_response(
            mock_fakesat_variables,
            {'HDF5_GLOBAL': {'short_name': 'FAKE99'}}
        )

    @patch('pymods.var_info.open_url')
    def test_var_info_short_name(self, mock_open_url):
        """ Ensure an instance of the VarInfo class is correctly initiated. """
        short_name = 'ATL03'

        test_attributes = [
            {'HDF5_GLOBAL': {'short_name': short_name}},
            {'NC_GLOBAL': {'short_name': short_name}},
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
                global_attributes = {'HDF5_GLOBAL': {'short_name': short_name}}
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
        self.assertEqual(dataset.global_attributes,
                         {'HDF5_GLOBAL': {'short_name': 'ATL03'}})

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

        expected_global_attributes = {
            'HDF5_GLOBAL': {'short_name': 'FAKE99'},
            'global_override': 'GLOBAL',
            'fakesat_global_supplement': 'fakesat value'
        }
        self.assertEqual(dataset.global_attributes, expected_global_attributes)
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


class TestVarInfoDmr(TestCase):
    """ A class for testing the VarInfo class with a `.dmr` URL. """

    @classmethod
    def setUpClass(cls):
        """ Set up properties of the class that do not need to be reset between
            tests.

        """
        cls.logger = Logger('VarInfo tests')
        cls.dmr_url = 'http://test.opendap.org/opendap/hyrax/user/granule.dmr'
        cls.config_file = 'tests/unit/data/test_config.yml'
        cls.namespace = 'namespace_string'

        cls.mock_dataset = MockResponse(
            200,
            (f'<{cls.namespace}Dataset>'
             f'  <{cls.namespace}Float64 name="/ancillary_one">'
             f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
             f'      <{cls.namespace}Value>/ancillary_one</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Float64>'
             f'  <{cls.namespace}Float64 name="/dimension_one">'
             f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
             f'      <{cls.namespace}Value>/dimension_one</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Float64>'
             f'  <{cls.namespace}Float64 name="/latitude">'
             f'    <{cls.namespace}Dim name="/dimension_one" />'
             f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
             f'      <{cls.namespace}Value>/latitude</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Float64>'
             f'  <{cls.namespace}Float64 name="/longitude">'
             f'    <{cls.namespace}Dim name="/dimension_one" />'
             f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
             f'      <{cls.namespace}Value>/longitude</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Float64>'
             f'  <{cls.namespace}Float64 name="/metadata_variable">'
             f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
             f'      <{cls.namespace}Value>/metadata_variable</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Float64>'
             f'  <{cls.namespace}Float64 name="/science_variable">'
             f'    <{cls.namespace}Dim name="/dimension_one" />'
             f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
             f'      <{cls.namespace}Value>/science_variable</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'    <{cls.namespace}Attribute name="ancillary_variables" type="String">'
             f'      <{cls.namespace}Value>/ancillary_one</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'    <{cls.namespace}Attribute name="coordinates" type="String">'
             f'      <{cls.namespace}Value>/latitude, /longitude</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'    <{cls.namespace}Attribute name="subset_control_variables" type="String">'
             f'      <{cls.namespace}Value>/subset_one</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Float64>'
             f'  <{cls.namespace}Float64 name="/subset_one">'
             f'    <{cls.namespace}Dim name="/dimension_one" />'
             f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
             f'      <{cls.namespace}Value>/subset_one</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'    <{cls.namespace}Attribute name="coordinates" type="String">'
             f'      <{cls.namespace}Value>/latitude, /longitude</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Float64>'
             f'  <{cls.namespace}Attribute name="HDF5_GLOBAL" type="Container">'
             f'    <{cls.namespace}Attribute name="short_name" type="String">'
             f'      <{cls.namespace}Value>ATL03</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Attribute>'
             f'</{cls.namespace}Dataset>')
        )

        cls.mock_dataset_two = MockResponse(
            200,
            (f'<{cls.namespace}Dataset>'
             f'  <{cls.namespace}Float64 name="/excude_one_has_coordinates">'
             f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
             f'      <{cls.namespace}Value>/exclude_one/has_coordinates</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'    <{cls.namespace}Attribute name="coordinates" type="String">'
             f'      <{cls.namespace}Value>../science/latitude, ../science/longitude</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Float64>'
             f'  <{cls.namespace}Float64 name="/required_group_has_no_coordinates">'
             f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
             f'      <{cls.namespace}Value>/required_group/has_no_coordinates</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Float64>'
             f'  <{cls.namespace}Float64 name="/science_interesting_thing">'
             f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
             f'      <{cls.namespace}Value>/science/interesting_thing</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'    <{cls.namespace}Attribute name="coordinates" type="String">'
             f'      <{cls.namespace}Value>latitude, longitude</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Float64>'
             f'  <{cls.namespace}Float64 name="/science_latitude">'
             f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
             f'      <{cls.namespace}Value>/science/latitude</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Float64>'
             f'  <{cls.namespace}Float64 name="/science_longitude">'
             f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
             f'      <{cls.namespace}Value>/science/longitude</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Float64>'
             f'  <{cls.namespace}Attribute name="HDF5_GLOBAL" type="Container">'
             f'    <{cls.namespace}Attribute name="short_name" type="String">'
             f'      <{cls.namespace}Value>FAKE99</{cls.namespace}Value>'
             f'    </{cls.namespace}Attribute>'
             f'  </{cls.namespace}Attribute>'
            f'</{cls.namespace}Dataset>')
        )

    @patch('pymods.var_info.requests.get')
    def test_var_info_instantiation_no_augmentation(self, mock_requests_get):
        """ Ensure VarInfo instantiates correctly, creating records of all the
            variables in the granule, and correctly deciding if they are
            science variables, metadata or references. This test uses a mission
            and short name that do not have any CF overrides or supplements..

        """
        mock_requests_get.return_value = self.mock_dataset
        dataset = VarInfo(self.dmr_url, self.logger,
                          config_file=self.config_file)

        self.assertEqual(dataset.short_name, 'ATL03')
        self.assertEqual(dataset.mission, 'ICESat2')
        self.assertEqual(dataset.global_attributes,
                         {'HDF5_GLOBAL': {'short_name': 'ATL03'}})

        self.assertEqual(set(dataset.metadata_variables.keys()),
                         {'/ancillary_one', '/dimension_one', '/latitude',
                          '/longitude', '/metadata_variable'})
        self.assertEqual(set(dataset.variables_with_coordinates.keys()),
                         {'/science_variable', '/subset_one'})
        self.assertEqual(dataset.references, {'/ancillary_one',
                                              '/dimension_one', '/latitude',
                                              '/longitude', '/subset_one'})

    @patch('pymods.var_info.requests.get')
    def test_var_info_instantiation_cf_augmentation(self, mock_requests_get):
        """ Ensure VarInfo instantiates correcly, using a missions that has
            overrides and supplements in the CFConfig class.

        """
        mock_requests_get.return_value = self.mock_dataset_two
        dataset = VarInfo(self.dmr_url, self.logger,
                          config_file=self.config_file)

        expected_global_attributes = {
            'HDF5_GLOBAL': {'short_name': 'FAKE99'},
            'global_override': 'GLOBAL',
            'fakesat_global_supplement': 'fakesat value'
        }
        self.assertEqual(dataset.global_attributes, expected_global_attributes)
        self.assertEqual(set(dataset.metadata_variables.keys()),
                         {'/science/latitude', '/science/longitude',
                          '/required_group/has_no_coordinates'})
        self.assertEqual(set(dataset.variables_with_coordinates.keys()),
                         {'/science/interesting_thing',
                          '/exclude_one/has_coordinates'})
        self.assertEqual(set(dataset.references), {'/science/latitude',
                                                   '/science/longitude'})

    @patch('pymods.var_info.requests.get')
    def test_var_info_request_error(self, mock_requests_get):
        """ Ensure VarInfo gracefully fails when a dataset object cannot be
            retrieved from OPeNDAP via an HTTP request.

        """
        mock_requests_get.side_effect = HTTPClientError('dmr problem')

        with self.assertRaises(HTTPClientError):
            VarInfo(self.dmr_url, self.logger)

    @patch('pymods.var_info.requests.get')
    def test_var_info_get_science_variables(self, mock_requests_get):
        """ Ensure the correct set of science variables is returned. This
            should account for excluded science variables defined in the
            associated instance of the `CFConfig` class.

        """
        mock_requests_get.return_value = self.mock_dataset_two
        dataset = VarInfo(self.dmr_url, self.logger,
                          config_file=self.config_file)

        science_variables = dataset.get_science_variables()
        self.assertEqual(science_variables, {'/science/interesting_thing'})

    @patch('pymods.var_info.requests.get')
    def test_var_info_get_metadata_variables(self, mock_requests_get):
        """ Ensure the correct set of metadata variables (those without
            coordinate references) is returned. This should exclude variables
            that are also referred to by others via the metadata such as the
            coordinates attribute.

            This set should also include science variables that are explicitly
            excluded by the `CFConfig` instance.

        """
        mock_requests_get.return_value = self.mock_dataset_two
        dataset = VarInfo(self.dmr_url, self.logger,
                          config_file=self.config_file)

        metadata_variables = dataset.get_metadata_variables()
        self.assertEqual(metadata_variables,
                         {'/required_group/has_no_coordinates',
                          '/exclude_one/has_coordinates'})

    @patch('pymods.var_info.requests.get')
    def test_var_info_get_required_variables(self, mock_requests_get):
        """ Ensure a full list of variables is returned when the VarInfo class
            is asked for those variables required to make a viable output
            granule. This should recursively search the references of all
            requested variables, to also include supporting variables such as
            coordinates, dimensions, ancillary_variables and
            subset_control_variables.

        """
        mock_requests_get.return_value = self.mock_dataset_two
        dataset = VarInfo(self.dmr_url, self.logger,
                          config_file=self.config_file)

        required_variables = dataset.get_required_variables(
            {'/science/interesting_thing'}
        )
        self.assertEqual(required_variables, {'/required_group/has_no_coordinates',
                                              '/science/interesting_thing',
                                              '/science/latitude',
                                              '/science/longitude'})


class TestVariablePydap(TestCase):
    """ Tests for the Variable class using `pydap.model.BaseType` input. """

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
            `BaseType` instance.

        """
        variable = Variable(self.pydap_variable, self.fakesat_config,
                            self.fakesat_map)
        self.assertEqual(variable.full_name_path, '/group/variable')
        self.assertEqual(variable.group_path, '/group')
        self.assertEqual(variable.name, 'variable')
        self.assertEqual(variable.attributes.get('units'), 'm')
        self.assertEqual(variable.ancillary_variables, {'/ancillary_data/epoch'})
        self.assertEqual(variable.coordinates, {'/group/latitude',
                                                '/group/longitude'})
        self.assertEqual(variable.dimensions, {'/dimensions'})
        self.assertEqual(variable.subset_control_variables,
                         {'/group/begin', '/group/count'})

    def test_variable_cf_override(self):
        """ Ensure a CF attribute is overridden by the `CFConfig` value. """
        variable_attributes = {'coordinates': 'latitude, longitude',
                               'fullnamepath': '/coordinates_group/science'}
        pydap_variable = BaseType('/coordinates_group/science', np.ones((2, 2)),
                                  attributes=variable_attributes)
        pydap_map = {'/coordinates_group/science': '/coordinates_group/science'}

        variable = Variable(pydap_variable, self.fakesat_config,
                            pydap_map)
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


class TestVariableDmr(TestCase):
    """ Tests for the Variable class using `xml.etree.ElementTree` input. """

    @classmethod
    def setUpClass(cls):
        """ Set up properties of the class that do not need to be reset between
            tests.

        """
        cls.config_file = 'tests/unit/data/test_config.yml'
        cls.fakesat_config = CFConfig('FakeSat', 'FAKE99',
                                      config_file=cls.config_file)
        cls.fakesat_map = {'/group_dimension': '/group/dimension',
                           '/group_variable': '/group/variable'}
        cls.namespace = 'namespace_string'
        cls.dmr_variable = ET.fromstring(
            f'<{cls.namespace}Float64 name="/group_variable">'
            f'  <{cls.namespace}Dim name="/group_dimension" />'
            f'  <{cls.namespace}Attribute name="ancillary_variables" type="String">'
            f'    <{cls.namespace}Value>/ancillary_data/epoch</{cls.namespace}Value>'
            f'  </{cls.namespace}Attribute>'
            f'  <{cls.namespace}Attribute name="coordinates" type="String">'
            f'    <{cls.namespace}Value>latitude, longitude</{cls.namespace}Value>'
            f'  </{cls.namespace}Attribute>'
            f'  <{cls.namespace}Attribute name="fullnamepath" type="String">'
            f'    <{cls.namespace}Value>/group/variable</{cls.namespace}Value>'
            f'  </{cls.namespace}Attribute>'
            f'  <{cls.namespace}Attribute name="subset_control_variables" type="String">'
            f'    <{cls.namespace}Value>begin count</{cls.namespace}Value>'
            f'  </{cls.namespace}Attribute>'
            f'  <{cls.namespace}Attribute name="units" type="String">'
            f'    <{cls.namespace}Value>m</{cls.namespace}Value>'
            f'  </{cls.namespace}Attribute>'
            f'</{cls.namespace}Float64>'
        )

    def test_variable_instantiation(self):
        """ Ensure a `Variable` instance can be created from an input `.dmr`
            XML element instance.

        """
        variable = Variable(self.dmr_variable, self.fakesat_config,
                            self.fakesat_map, self.namespace)
        self.assertEqual(variable.full_name_path, '/group/variable')
        self.assertEqual(variable.group_path, '/group')
        self.assertEqual(variable.name, 'variable')
        self.assertEqual(variable.attributes.get('units'), 'm')
        self.assertEqual(variable.ancillary_variables, {'/ancillary_data/epoch'})
        self.assertEqual(variable.coordinates, {'/group/latitude',
                                                '/group/longitude'})
        self.assertEqual(variable.dimensions, {'/group/dimension'})
        self.assertEqual(variable.subset_control_variables,
                         {'/group/begin', '/group/count'})

    def test_variable_cf_override(self):
        """ Ensure a CF attribute is overridden by the `CFConfig` value. """
        dmr_variable = ET.fromstring(
            f'<{self.namespace}Float64 name="/coordinates_group_science">'
            f'  <{self.namespace}Attribute name="coordinates" type="String">'
            f'    <{self.namespace}Value>latitude, longitude</{self.namespace}Value>'
            f'  </{self.namespace}Attribute>'
            f'  <{self.namespace}Attribute name="fullnamepath" type="String">'
            f'    <{self.namespace}Value>/coordinates_group/science</{self.namespace}Value>'
            f'  </{self.namespace}Attribute>'
            f'</{self.namespace}Float64>'
        )

        name_map = {'/coordinates_group_science': '/coordinates_group/science'}

        variable = Variable(dmr_variable, self.fakesat_config, name_map,
                            self.namespace)
        self.assertEqual(variable.coordinates, {'/coordinates_group/lat',
                                                '/coordinates_group/lon'})

    def test_variable_reference_qualification(self):
        """ Ensure different reference types (relative, absolute) are correctly
            qualified.

        """
        name_map = {'/gt1r_heights_bckgrd_mean': '/gt1r/heights/bckgrd_mean',
                    '/gt1r_latitude': '/gt1r/latitude',
                    '/gt1r_longitude': '/gt1r/longitude',
                    '/latitude': '/latitude',
                    '/longitude': '/longitude'}

        variable_name = '/gt1r/heights/bckgrd_mean'
        test_args = [['In parent group', '../latitude', '/gt1r/latitude'],
                     ['In granule root', '/latitude', '/latitude'],
                     ['Relative in same', './latitude', '/gt1r/heights/latitude'],
                     ['Basename only', 'latitude', '/gt1r/heights/latitude']]

        for description, coordinates, qualified_reference in test_args:
            with self.subTest(description):
                dmr_variable = ET.fromstring(
                    f'<{self.namespace}Float64 name="/gt1r_heights_bckgrd_mean">'
                    f'  <{self.namespace}Attribute name="coordinates" type="String">'
                    f'    <{self.namespace}Value>{coordinates}</{self.namespace}Value>'
                    f'  </{self.namespace}Attribute>'
                    f'  <{self.namespace}Attribute name="fullnamepath" type="String">'
                    f'    <{self.namespace}Value>{variable_name}</{self.namespace}Value>'
                    f'  </{self.namespace}Attribute>'
                    f'</{self.namespace}Float64>'
                )
                variable = Variable(dmr_variable, self.fakesat_config,
                                    name_map, self.namespace)
                self.assertEqual(variable.coordinates, {qualified_reference})

    def test_variable_get_references(self):
        """ Ensure that a set of absolute paths to all variables referred to
            in the ancillary_variables, coordinates, dimensions and
            subset_control_variables is returned.

        """
        variable = Variable(self.dmr_variable, self.fakesat_config,
                            self.fakesat_map, self.namespace)

        references = variable.get_references()

        self.assertEqual(references, {'/ancillary_data/epoch',
                                      '/group/latitude',
                                      '/group/longitude',
                                      '/group/dimension',
                                      '/group/begin',
                                      '/group/count'})

    def test_dmr_dimension_conversion(self):
        """ Ensure that if a dimension has a `.dmr` style name it is converted
            to the full path, for example:

            /group_one_group_two_variable

            becomes:

            /group_one/group_two/variable

        """
        variable_name = '/group_one_variable'
        name_map = {'/group_one_delta_time': '/group_one/delta_time',
                    '/group_one_variable': '/group_one/variable'}

        dmr_variable = ET.fromstring(
            f'<{self.namespace}Float64 name="{variable_name}">'
            f'  <{self.namespace}Dim name="/group_one_delta_time" />'
            f'  <{self.namespace}Attribute name="fullnamepath" type="String">'
            f'    <{self.namespace}Value>{variable_name}</{self.namespace}Value>'
            f'  </{self.namespace}Attribute>'
            f'</{self.namespace}Float64>'
        )
        variable = Variable(dmr_variable, self.fakesat_config, name_map,
                            self.namespace)

        self.assertEqual(variable.dimensions, {'/group_one/delta_time'})
