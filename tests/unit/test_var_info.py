from logging import Logger
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import patch
import re

from harmony.util import config
from pymods.var_info import VarInfo
from tests.utilities import write_dmr


class TestVarInfo(TestCase):
    """ A class for testing the `VarInfo` class with a `.dmr` URL. """

    @classmethod
    def setUpClass(cls):
        """ Set up properties of the class that do not need to be reset between
            tests.

        """
        cls.logger = Logger('VarInfo tests')
        cls.dmr_url = 'http://test.opendap.org/opendap/hyrax/user/granule.dmr'
        cls.config_file = 'tests/unit/data/test_config.yml'
        cls.env_config = config(False)
        cls.access_token = 'fake-token'

        cls.namespace = 'namespace_string'

        cls.mock_dataset = (
            f'<Dataset xmlns="{cls.namespace}">'
            '  <Float64 name="ancillary_one">'
            '  </Float64>'
            '  <Float64 name="dimension_one">'
            '  </Float64>'
            '  <Float64 name="latitude">'
            '    <Dim name="/dimension_one" />'
            '  </Float64>'
            '  <Float64 name="longitude">'
            '    <Dim name="/dimension_one" />'
            '  </Float64>'
            '  <Float64 name="metadata_variable">'
            '  </Float64>'
            '  <Float64 name="science_variable">'
            '    <Dim name="/dimension_one" />'
            '    <Attribute name="ancillary_variables" type="String">'
            '      <Value>/ancillary_one</Value>'
            '    </Attribute>'
            '    <Attribute name="coordinates" type="String">'
            '      <Value>/latitude, /longitude</Value>'
            '    </Attribute>'
            '    <Attribute name="subset_control_variables" type="String">'
            '      <Value>/subset_one</Value>'
            '    </Attribute>'
            '  </Float64>'
            '  <Float64 name="subset_one">'
            '    <Dim name="/dimension_one" />'
            '    <Attribute name="coordinates" type="String">'
            '      <Value>/latitude, /longitude</Value>'
            '    </Attribute>'
            '  </Float64>'
            '  <Group name="METADATA">'
            '    <Group name="DatasetIdentification">'
            '      <Attribute name="shortName">'
            '        <Value>ATL03</Value>'
            '      </Attribute>'
            '    </Group>'
            '  </Group>'
            '</Dataset>'
        )

        cls.mock_dataset_two = (
            f'<Dataset xmlns="{cls.namespace}">'
            '  <Group name="exclude_one">'
            '    <Float64 name="has_coordinates">'
            '      <Attribute name="coordinates" type="String">'
            '        <Value>../science/latitude, ../science/longitude</Value>'
            '      </Attribute>'
            '    </Float64>'
            '  </Group>'
            '  <Group name="required_group">'
            '    <Float64 name="has_no_coordinates">'
            '    </Float64>'
            '  </Group>'
            '  <Group name="science">'
            '    <Float64 name="interesting_thing">'
            '      <Attribute name="coordinates" type="String">'
            '        <Value>latitude, longitude</Value>'
            '      </Attribute>'
            '    </Float64>'
            '    <Float64 name="latitude">'
            '    </Float64>'
            '    <Float64 name="longitude">'
            '    </Float64>'
            '  </Group>'
            '  <Group name="METADATA">'
            '    <Group name="DatasetIdentification">'
            '      <Attribute name="shortName" type="String">'
            '        <Value>FAKE99</Value>'
            '      </Attribute>'
            '    </Group>'
            '  </Group>'
            '</Dataset>'
        )

    def setUp(self):
        self.output_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.output_dir)

    @patch('pymods.var_info.download_url')
    def test_var_info_short_name(self, mock_download_url):
        """ Ensure an instance of the VarInfo class correctly identifies a
            collection short name if it is stored as a metadata attribute in
            any of the prescribed locations.

        """
        short_name = 'ATL03'

        hdf5_global = (
            '  <Group name="HDF5_GLOBAL">'
            '    <Attribute name="short_name">'
            f'      <Value>{short_name}</Value>'
            '    </Attribute>'
            '  </Group>'
        )

        nc_global = (
            '  <Group name="NC_GLOBAL">'
            '    <Attribute name="short_name">'
            f'      <Value>{short_name}</Value>'
            '    </Attribute>'
            '  </Group>'
        )

        metadata_dataset_lower_case = (
            '  <Group name="Metadata">'
            '    <Group name="DatasetIdentification">'
            '      <Attribute name="shortName">'
            f'        <Value>{short_name}</Value>'
            '      </Attribute>'
            '    </Group>'
            '  </Group>'
        )

        metadata_dataset_upper_case = (
            '  <Group name="METADATA">'
            '    <Group name="DatasetIdentification">'
            '      <Attribute name="shortName">'
            f'        <Value>{short_name}</Value>'
            '      </Attribute>'
            '    </Group>'
            '  </Group>'
        )

        metadata_series_lower_case = (
            '  <Group name="Metadata">'
            '    <Group name="SeriesIdentification">'
            '      <Attribute name="shortName">'
            f'        <Value>{short_name}</Value>'
            '      </Attribute>'
            '    </Group>'
            '  </Group>'
        )

        metadata_series_upper_case = (
            '  <Group name="METADATA">'
            '    <Group name="SeriesIdentification">'
            '      <Attribute name="shortName">'
            f'        <Value>{short_name}</Value>'
            '      </Attribute>'
            '    </Group>'
            '  </Group>'
        )

        root_group_short_name = (
            '  <Attribute name="short_name">'
            f'    <Value>{short_name}</Value>'
            '  </Attribute>'
        )

        test_attributes = [hdf5_global, nc_global, metadata_dataset_lower_case,
                           metadata_dataset_upper_case,
                           metadata_series_lower_case,
                           metadata_series_upper_case, root_group_short_name]

        for global_attributes in test_attributes:
            with self.subTest(global_attributes):
                mock_dmr = (f'<Dataset xmlns="{self.namespace}">'
                            f'{global_attributes}'
                            '</Dataset>')
                mock_download_url.side_effect = [write_dmr(self.output_dir,
                                                           mock_dmr)]

                dataset = VarInfo(self.dmr_url,
                                  self.logger,
                                  self.output_dir,
                                  self.access_token,
                                  self.env_config)

                mock_download_url.assert_called_once_with(self.dmr_url,
                                                          self.output_dir,
                                                          self.logger,
                                                          self.access_token,
                                                          self.env_config)
                self.assertEqual(dataset.short_name, short_name)

            mock_download_url.reset_mock()

        with self.subTest('No short name'):
            mock_dmr = (f'<Dataset xmlns="{self.namespace}"></Dataset>')
            mock_download_url.side_effect = [write_dmr(self.output_dir,
                                                       mock_dmr)]
            dataset = VarInfo(self.dmr_url,
                              self.logger,
                              self.output_dir,
                              self.access_token,
                              self.env_config)

            mock_download_url.assert_called_once_with(self.dmr_url,
                                                      self.output_dir,
                                                      self.logger,
                                                      self.access_token,
                                                      self.env_config)
            self.assertEqual(dataset.short_name, None)
            mock_download_url.reset_mock()

    @patch('pymods.var_info.download_url')
    def test_var_info_mission(self, mock_download_url):
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
                mock_dmr = (f'<Dataset xmlns="{self.namespace}">'
                            '  <Attribute name="short_name">'
                            f'    <Value>{short_name}</Value>'
                            '  </Attribute>'
                            '</Dataset>')
                mock_download_url.side_effect = [write_dmr(self.output_dir,
                                                           mock_dmr)]

                dataset = VarInfo(self.dmr_url,
                                  self.logger,
                                  self.output_dir,
                                  self.access_token,
                                  self.env_config)

                mock_download_url.assert_called_once_with(self.dmr_url,
                                                          self.output_dir,
                                                          self.logger,
                                                          self.access_token,
                                                          self.env_config)
                self.assertEqual(dataset.mission, expected_mission)

            mock_download_url.reset_mock()


    @patch('pymods.var_info.download_url')
    def test_var_info_instantiation_no_augmentation(self, mock_download_url):
        """ Ensure VarInfo instantiates correctly, creating records of all the
            variables in the granule, and correctly deciding if they are
            science variables, metadata or references. This test uses a mission
            and short name that do not have any CF overrides or supplements.

        """
        mock_download_url.side_effect = [write_dmr(self.output_dir,
                                                   self.mock_dataset)]
        dataset = VarInfo(self.dmr_url, self.logger, self.output_dir,
                          self.access_token, self.env_config,
                          config_file=self.config_file)

        self.assertEqual(dataset.short_name, 'ATL03')
        self.assertEqual(dataset.mission, 'ICESat2')
        self.assertEqual(
            dataset.global_attributes,
            {'METADATA': {'DatasetIdentification': {'shortName': 'ATL03'}}}
        )

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
        """ Ensure VarInfo instantiates correcly, using a missions that has
            overrides and supplements in the CFConfig class.

        """
        mock_download_url.side_effect = [write_dmr(self.output_dir,
                                                   self.mock_dataset_two)]
        dataset = VarInfo(self.dmr_url, self.logger, self.output_dir,
                          self.access_token, self.env_config,
                          config_file=self.config_file)

        expected_global_attributes = {
            'METADATA': {'DatasetIdentification': {'shortName': 'FAKE99'}},
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
        dataset = VarInfo(self.dmr_url, self.logger, self.output_dir,
                          self.access_token, self.env_config,
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
        dataset = VarInfo(self.dmr_url, self.logger, self.output_dir,
                          self.access_token, self.env_config,
                          config_file=self.config_file)

        metadata_variables = dataset.get_metadata_variables()
        self.assertEqual(metadata_variables,
                         {'/required_group/has_no_coordinates',
                          '/exclude_one/has_coordinates'})

    @patch('pymods.var_info.download_url')
    def test_var_info_get_required_variables(self, mock_download_url):
        """ Ensure a full list of variables is returned when the VarInfo
            class is asked for those variables required to make a viable output
            granule. This should recursively search the references of all
            requested variables, to also include supporting variables such as
            coordinates, dimensions, ancillary_variables and
            subset_control_variables.

        """
        mock_download_url.side_effect = [write_dmr(self.output_dir,
                                                   self.mock_dataset_two)]
        dataset = VarInfo(self.dmr_url, self.logger, self.output_dir,
                          self.access_token, self.env_config,
                          config_file=self.config_file)

        required_variables = dataset.get_required_variables(
            {'/science/interesting_thing'}
        )
        self.assertEqual(required_variables, {'/required_group/has_no_coordinates',
                                              '/science/interesting_thing',
                                              '/science/latitude',
                                              '/science/longitude'})

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
                result = VarInfo.variable_is_excluded(variable_name,
                                                      re_pattern)

                self.assertEqual(result, expected_result)

    def test_exclude_fake_dimensions(self):
        """ Ensure a set of required variables will not include any dimension
            generated by OPeNDAP, that does not actually exist in a granule.
            Only variables with names like FakeDim0, FakeDim1, etc should be
            removed.

        """
        input_variables = {'/science_variable', '/FakeDim0', '/other_science',
                           '/FakeDim1234', '/nested/FakeDim0'}

        required_variables = VarInfo.exclude_fake_dimensions(input_variables)

        self.assertEqual(required_variables,
                         {'/science_variable', '/other_science'})
