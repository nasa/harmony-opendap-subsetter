from logging import Logger
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import patch
import re

from webob.exc import HTTPClientError

from pymods.exceptions import PydapRetrievalError
from pymods.var_info import VarInfoFromDmr, VarInfoFromPydap
from tests.utilities import (generate_pydap_response, mock_fakesat_variables,
                             mock_variables, write_dmr)


class TestVarInfoFromPydap(TestCase):
    """ A class for testing the `VarInfoFromPydap` class. """

    @classmethod
    def setUpClass(cls):
        """ Set up properties of the class that do not need to be reset between
            tests.

        """
        cls.logger = Logger('VarInfoFromPydap tests')
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

    def setUp(self):
        self.output_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.output_dir)

    @patch('pymods.var_info.open_url')
    def test_var_info_short_name(self, mock_open_url):
        """ Ensure an instance of the VarInfoFromPydap class is correctly
            initiated.

        """
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
                dataset = VarInfoFromPydap(self.pydap_url, self.logger,
                                           self.output_dir)

                mock_open_url.assert_called_once_with(self.pydap_url)
                self.assertEqual(dataset.short_name, short_name)

            mock_open_url.reset_mock()

        with self.subTest('No short name'):
            mock_response = generate_pydap_response({'sea_surface_temp': {}},
                                                    {})
            mock_open_url.return_value = mock_response
            dataset = VarInfoFromPydap(self.pydap_url, self.logger,
                                       self.output_dir)

            mock_open_url.assert_called_once_with(self.pydap_url)
            self.assertEqual(dataset.short_name, None)
            mock_open_url.reset_mock()

    @patch('pymods.var_info.open_url')
    def test_var_info_mission(self, mock_open_url):
        """ Ensure VarInfoFromPydap can identify the correct mission given a
            collection short name, or absence of one.

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
                dataset = VarInfoFromPydap(self.pydap_url, self.logger,
                                           self.output_dir)

                mock_open_url.assert_called_once_with(self.pydap_url)
                self.assertEqual(dataset.mission, expected_mission)

            mock_open_url.reset_mock()

    @patch('pymods.var_info.open_url')
    def test_var_info_instantiation_no_augmentation(self, mock_open_url):
        """ Ensure VarInfoFromPydap instantiates correctly, creating records of
            all the variables in the granule, and correctly deciding if they
            are science variables, metadata or references. This test uses a
            mission and short name that do not have any CF overrides or
            supplements.

        """
        mock_open_url.return_value = self.mock_dataset
        dataset = VarInfoFromPydap(self.pydap_url, self.logger,
                                   self.output_dir,
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
        self.assertEqual(dataset.references, {'/ancillary_one',
                                              '/dimension_one', '/latitude',
                                              '/longitude', '/subset_one'})

    @patch('pymods.var_info.open_url')
    def test_var_info_instantiation_cf_augmentation(self, mock_open_url):
        """ Ensure VarInfoFromPydap instantiates correcly, using a missions
            that has overrides and supplements in the CFConfig class.

        """
        mock_open_url.return_value = self.mock_dataset_two
        dataset = VarInfoFromPydap(self.pydap_url, self.logger,
                                   self.output_dir,
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
        """ Ensure VarInfoFromPydap gracefully fails when a dataset object
            cannot be retrieved from pydap.

        """
        mock_open_url.side_effect = HTTPClientError('pydap problem')

        with self.assertRaises(PydapRetrievalError):
            VarInfoFromPydap(self.pydap_url, self.logger, self.output_dir)

    @patch('pymods.var_info.open_url')
    def test_var_info_get_science_variables(self, mock_open_url):
        """ Ensure the correct set of science variables is returned. This
            should account for excluded science variables defined in the
            associated instance of the CFConfig class.

        """
        mock_open_url.return_value = self.mock_dataset_two
        dataset = VarInfoFromPydap(self.pydap_url, self.logger,
                                   self.output_dir,
                                   config_file=self.config_file)

        science_variables = dataset.get_science_variables()
        self.assertEqual(science_variables, {'/science/interesting_thing'})

    def test_var_info_variable_is_excluded(self):
        """ Ensure the a variable is correctly identified as being excluded or
            not, including when there are not exclusions for the collection.

        """
        variable = 'variable_name'
        test_args = [['No exclusions', '', variable, False],
                     ['Not excluded', 'not_var', variable, False],
                     ['Excluded', 'var', variable, True]]

        for description, pattern, variable_name, expected_result in test_args:
            with self.subTest(description):
                re_pattern = re.compile(pattern)
                result = VarInfoFromPydap.variable_is_excluded(variable_name,
                                                               re_pattern)
                self.assertEqual(result, expected_result)

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
        dataset = VarInfoFromPydap(self.pydap_url, self.logger,
                                   self.output_dir,
                                   config_file=self.config_file)

        metadata_variables = dataset.get_metadata_variables()
        self.assertEqual(metadata_variables,
                         {'/required_group/has_no_coordinates',
                          '/exclude_one/has_coordinates'})

    @patch('pymods.var_info.open_url')
    def test_var_info_get_required_variables(self, mock_open_url):
        """ Ensure a full list of variables is returned when the
            VarInfoFromPydap class is asked for those variables required to
            make a viable output granule. This should recursively search the
            references of all requested variables, to also include supporting
            variables such as coordinates, dimensions, ancillary_variables and
            subset_control_variables.

        """
        mock_open_url.return_value = self.mock_dataset_two
        dataset = VarInfoFromPydap(self.pydap_url, self.logger,
                                   self.output_dir,
                                   config_file=self.config_file)

        required_variables = dataset.get_required_variables(
            {'/science/interesting_thing'}
        )
        self.assertEqual(required_variables, {'/required_group/has_no_coordinates',
                                              '/science/interesting_thing',
                                              '/science/latitude',
                                              '/science/longitude'})

    def test_exclude_fake_dimensions(self):
        """ Ensure a set of required variables will not include any dimension
            generated by OPeNDAP, that does not actually exist in a granule.
            Only variables with names like FakeDim0, FakeDim1, etc should be
            removed.

        """
        input_variables = {'/science_variable', '/FakeDim0', '/other_science',
                           '/FakeDim1234', '/nested/FakeDim0'}

        required_variables = VarInfoFromPydap.exclude_fake_dimensions(
            input_variables
        )

        self.assertEqual(required_variables,
                         {'/science_variable', '/other_science'})

class TestVarInfoDmr(TestCase):
    """ A class for testing the `VarInfoFromDmr` class with a `.dmr` URL. """

    @classmethod
    def setUpClass(cls):
        """ Set up properties of the class that do not need to be reset between
            tests.

        """
        cls.logger = Logger('VarInfoFromDmr tests')
        cls.dmr_url = 'http://test.opendap.org/opendap/hyrax/user/granule.dmr'
        cls.config_file = 'tests/unit/data/test_config.yml'
        cls.namespace = 'namespace_string'

        cls.mock_dataset = (
            f'<{cls.namespace}Dataset>'
            f'  <{cls.namespace}Float64 name="ancillary_one">'
            f'  </{cls.namespace}Float64>'
            f'  <{cls.namespace}Float64 name="dimension_one">'
            f'  </{cls.namespace}Float64>'
            f'  <{cls.namespace}Float64 name="latitude">'
            f'    <{cls.namespace}Dim name="/dimension_one" />'
            f'  </{cls.namespace}Float64>'
            f'  <{cls.namespace}Float64 name="longitude">'
            f'    <{cls.namespace}Dim name="/dimension_one" />'
            f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
            f'      <{cls.namespace}Value>/longitude</{cls.namespace}Value>'
            f'    </{cls.namespace}Attribute>'
            f'  </{cls.namespace}Float64>'
            f'  <{cls.namespace}Float64 name="metadata_variable">'
            f'    <{cls.namespace}Attribute name="fullnamepath" type="String">'
            f'      <{cls.namespace}Value>/metadata_variable</{cls.namespace}Value>'
            f'    </{cls.namespace}Attribute>'
            f'  </{cls.namespace}Float64>'
            f'  <{cls.namespace}Float64 name="science_variable">'
            f'    <{cls.namespace}Dim name="/dimension_one" />'
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
            f'  <{cls.namespace}Float64 name="subset_one">'
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
            f'</{cls.namespace}Dataset>'
        )

        cls.mock_dataset_two = (
            f'<{cls.namespace}Dataset>'
            f'  <{cls.namespace}Group name="exclude_one">'
            f'    <{cls.namespace}Float64 name="has_coordinates">'
            f'      <{cls.namespace}Attribute name="coordinates" type="String">'
            f'        <{cls.namespace}Value>../science/latitude, ../science/longitude</{cls.namespace}Value>'
            f'      </{cls.namespace}Attribute>'
            f'    </{cls.namespace}Float64>'
            f'  </{cls.namespace}Group>'
            f'  <{cls.namespace}Group name="required_group">'
            f'    <{cls.namespace}Float64 name="has_no_coordinates">'
            f'    </{cls.namespace}Float64>'
            f'  </{cls.namespace}Group>'
            f'  <{cls.namespace}Group name="science">'
            f'    <{cls.namespace}Float64 name="interesting_thing">'
            f'      <{cls.namespace}Attribute name="coordinates" type="String">'
            f'        <{cls.namespace}Value>latitude, longitude</{cls.namespace}Value>'
            f'      </{cls.namespace}Attribute>'
            f'    </{cls.namespace}Float64>'
            f'    <{cls.namespace}Float64 name="latitude">'
            f'    </{cls.namespace}Float64>'
            f'    <{cls.namespace}Float64 name="longitude">'
            f'    </{cls.namespace}Float64>'
            f'  </{cls.namespace}Group>'
            f'  <{cls.namespace}Attribute name="HDF5_GLOBAL" type="Container">'
            f'    <{cls.namespace}Attribute name="short_name" type="String">'
            f'      <{cls.namespace}Value>FAKE99</{cls.namespace}Value>'
            f'    </{cls.namespace}Attribute>'
            f'  </{cls.namespace}Attribute>'
            f'</{cls.namespace}Dataset>'
        )

    def setUp(self):
        self.output_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.output_dir)

    @patch('pymods.var_info.download_url')
    def test_var_info_instantiation_no_augmentation(self, mock_download_url):
        """ Ensure VarInfoFromDmr instantiates correctly, creating records of
            all the variables in the granule, and correctly deciding if they
            are science variables, metadata or references. This test uses a
            mission and short name that do not have any CF overrides or
            supplements.

        """
        mock_download_url.side_effect = [write_dmr(self.output_dir,
                                                   self.mock_dataset)]
        dataset = VarInfoFromDmr(self.dmr_url, self.logger, self.output_dir,
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

    @patch('pymods.var_info.download_url')
    def test_var_info_instantiation_cf_augmentation(self, mock_download_url):
        """ Ensure VarInfoFromDmr instantiates correcly, using a missions that
            has overrides and supplements in the CFConfig class.

        """
        mock_download_url.side_effect = [write_dmr(self.output_dir,
                                                   self.mock_dataset_two)]
        dataset = VarInfoFromDmr(self.dmr_url, self.logger, self.output_dir,
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

    @patch('pymods.var_info.download_url')
    def test_var_info_get_science_variables(self, mock_download_url):
        """ Ensure the correct set of science variables is returned. This
            should account for excluded science variables defined in the
            associated instance of the `CFConfig` class.

        """
        mock_download_url.side_effect = [write_dmr(self.output_dir,
                                                   self.mock_dataset_two)]
        dataset = VarInfoFromDmr(self.dmr_url, self.logger, self.output_dir,
                                 config_file=self.config_file)

        science_variables = dataset.get_science_variables()
        self.assertEqual(science_variables, {'/science/interesting_thing'})

    @patch('pymods.var_info.download_url')
    def test_var_info_get_metadata_variables(self, mock_download_url):
        """ Ensure the correct set of metadata variables (those without
            coordinate references) is returned. This should exclude variables
            that are also referred to by others via the metadata such as the
            coordinates attribute.

            This set should also include science variables that are explicitly
            excluded by the `CFConfig` instance.

        """
        mock_download_url.side_effect = [write_dmr(self.output_dir,
                                                   self.mock_dataset_two)]
        dataset = VarInfoFromDmr(self.dmr_url, self.logger, self.output_dir,
                                 config_file=self.config_file)

        metadata_variables = dataset.get_metadata_variables()
        self.assertEqual(metadata_variables,
                         {'/required_group/has_no_coordinates',
                          '/exclude_one/has_coordinates'})

    @patch('pymods.var_info.download_url')
    def test_var_info_get_required_variables(self, mock_download_url):
        """ Ensure a full list of variables is returned when the VarInfoFromDmr
            class is asked for those variables required to make a viable output
            granule. This should recursively search the references of all
            requested variables, to also include supporting variables such as
            coordinates, dimensions, ancillary_variables and
            subset_control_variables.

        """
        mock_download_url.side_effect = [write_dmr(self.output_dir,
                                                   self.mock_dataset_two)]
        dataset = VarInfoFromDmr(self.dmr_url, self.logger, self.output_dir,
                                 config_file=self.config_file)

        required_variables = dataset.get_required_variables(
            {'/science/interesting_thing'}
        )
        self.assertEqual(required_variables, {'/required_group/has_no_coordinates',
                                              '/science/interesting_thing',
                                              '/science/latitude',
                                              '/science/longitude'})
