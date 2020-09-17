from typing import Dict
from unittest import TestCase
import xml.etree.ElementTree as ET

import numpy as np
from pydap.model import BaseType, DatasetType

from pymods.cf_config import CFConfig
from pymods.variable import VariableFromDmr, VariableFromPydap


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


class TestVariableFromPydap(TestCase):
    """ Tests for the `VariableFromPydap` class using `pydap.model.BaseType`
        input.

    """

    @classmethod
    def setUpClass(cls):
        """ Set up properties of the class that do not need to be reset between
            tests.

        """
        cls.config_file = 'tests/unit/data/test_config.yml'
        cls.fakesat_config = CFConfig('FakeSat', 'FAKE99',
                                      config_file=cls.config_file)
        cls.fakesat_map = {'/dimensions': 'dimensions'}
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
        variable = VariableFromPydap(self.pydap_variable, self.fakesat_config,
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

        variable = VariableFromPydap(pydap_variable, self.fakesat_config,
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
                variable = VariableFromPydap(pydap_variable,
                                             self.fakesat_config,
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

        variable = VariableFromPydap(pydap_variable, self.fakesat_config,
                                     pydap_map)
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
        pydap_map = {'/group_one/delta_time': 'group_one_delta_time',
                     '/group_one/variable': 'group_one_variable'}
        pydap_attributes = {'fullnamepath': variable_name}
        pydap_variable = BaseType(variable_name, np.ones((2, 2)),
                                  attributes=pydap_attributes,
                                  dimensions=('group_one_delta_time',))
        variable = VariableFromPydap(pydap_variable, self.fakesat_config,
                                     pydap_map)

        self.assertEqual(variable.dimensions, {'/group_one/delta_time'})


class TestVariableDmr(TestCase):
    """ Tests for the `VariableFromDmr` class using `xml.etree.ElementTree`
        input.

    """

    @classmethod
    def setUpClass(cls):
        """ Set up properties of the class that do not need to be reset between
            tests.

        """
        cls.config_file = 'tests/unit/data/test_config.yml'
        cls.fakesat_config = CFConfig('FakeSat', 'FAKE99',
                                      config_file=cls.config_file)
        cls.fakesat_map = {'/group/dimension': '/group_dimension',
                           '/group/variable': '/group_variable'}
        cls.namespace = 'namespace_string'
        cls.dmr_variable = ET.fromstring(
            f'<{cls.namespace}Float64 name="variable">'
            f'  <{cls.namespace}Dim name="dimension" />'
            f'  <{cls.namespace}Attribute name="ancillary_variables" type="String">'
            f'    <{cls.namespace}Value>/ancillary_data/epoch</{cls.namespace}Value>'
            f'  </{cls.namespace}Attribute>'
            f'  <{cls.namespace}Attribute name="coordinates" type="String">'
            f'    <{cls.namespace}Value>latitude, longitude</{cls.namespace}Value>'
            f'  </{cls.namespace}Attribute>'
            f'  <{cls.namespace}Attribute name="subset_control_variables" type="String">'
            f'    <{cls.namespace}Value>begin count</{cls.namespace}Value>'
            f'  </{cls.namespace}Attribute>'
            f'  <{cls.namespace}Attribute name="units" type="String">'
            f'    <{cls.namespace}Value>m</{cls.namespace}Value>'
            f'  </{cls.namespace}Attribute>'
            f'</{cls.namespace}Float64>'
        )
        cls.dmr_variable_path = '/group/variable'

    def test_variable_instantiation(self):
        """ Ensure a `Variable` instance can be created from an input `.dmr`
            XML element instance.

        """
        variable = VariableFromDmr(self.dmr_variable, self.fakesat_config,
                                   self.fakesat_map, self.namespace,
                                   self.dmr_variable_path)
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
            f'<{self.namespace}Float64 name="science">'
            f'  <{self.namespace}Attribute name="coordinates" type="String">'
            f'    <{self.namespace}Value>latitude, longitude</{self.namespace}Value>'
            f'  </{self.namespace}Attribute>'
            f'  <{self.namespace}Attribute name="fullnamepath" type="String">'
            f'    <{self.namespace}Value>/coordinates_group/science</{self.namespace}Value>'
            f'  </{self.namespace}Attribute>'
            f'</{self.namespace}Float64>'
        )

        name_map = {'/coordinates_group/science': '/coordinates_group_science'}

        variable = VariableFromDmr(dmr_variable, self.fakesat_config, name_map,
                                   self.namespace, '/coordinates_group/science')
        self.assertEqual(variable.coordinates, {'/coordinates_group/lat',
                                                '/coordinates_group/lon'})

    def test_variable_reference_qualification(self):
        """ Ensure different reference types (relative, absolute) are correctly
            qualified.

        """
        name_map = {'/gt1r/heights/bckgrd_mean': '/gt1r_heights_bckgrd_mean',
                    '/gt1r/latitude': '/gt1r_latitude',
                    '/gt1r/longitude': '/gt1r_longitude',
                    '/latitude': '/latitude',
                    '/longitude': '/longitude',
                    '/global_aerosol_frac': '/global_aerosol_frac',
                    '/global_lat': '/global_lat'}

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
                variable = VariableFromDmr(dmr_variable, self.fakesat_config,
                                           name_map, self.namespace)
                self.assertEqual(variable.coordinates, {qualified_reference})

        root_var_name = '/global_aerosol_frac'
        test_args = [
            ['Root, relative with leading slash', '/global_lat', '/global_lat'],
            ['Root, relative needs leading slash', 'global_lat', '/global_lat']
        ]

        for description, coordinates, qualified_reference in test_args:
            with self.subTest(description):
                dmr_variable = ET.fromstring(
                    f'<{self.namespace}Float64 name="/global_aerosol_frac">'
                    f'  <{self.namespace}Attribute name="coordinates" type="String">'
                    f'    <{self.namespace}Value>{coordinates}</{self.namespace}Value>'
                    f'  </{self.namespace}Attribute>'
                    f'  <{self.namespace}Attribute name="fullnamepath" type="String">'
                    f'    <{self.namespace}Value>{root_var_name}</{self.namespace}Value>'
                    f'  </{self.namespace}Attribute>'
                    f'</{self.namespace}Float64>'
                )

                variable = VariableFromDmr(dmr_variable, self.fakesat_config,
                                           name_map, self.namespace)
                self.assertEqual(variable.coordinates, {qualified_reference})

    def test_variable_get_references(self):
        """ Ensure that a set of absolute paths to all variables referred to
            in the ancillary_variables, coordinates, dimensions and
            subset_control_variables is returned.

        """
        variable = VariableFromDmr(self.dmr_variable, self.fakesat_config,
                                   self.fakesat_map, self.namespace,
                                   self.dmr_variable_path)

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
        name_map = {'/group_one/delta_time': '/group_one_delta_time',
                    '/group_one/variable': '/group_one_variable'}

        dmr_variable = ET.fromstring(
            f'<{self.namespace}Float64 name="{variable_name}">'
            f'  <{self.namespace}Dim name="/group_one_delta_time" />'
            f'  <{self.namespace}Attribute name="fullnamepath" type="String">'
            f'    <{self.namespace}Value>{variable_name}</{self.namespace}Value>'
            f'  </{self.namespace}Attribute>'
            f'</{self.namespace}Float64>'
        )
        variable = VariableFromDmr(dmr_variable, self.fakesat_config, name_map,
                                   self.namespace)

        self.assertEqual(variable.dimensions, {'/group_one/delta_time'})
