""" This module contains a class designed to read information from a `.dmr`
    file. This should group the input into science variables, metadata,
    coordinates, dimensions and ancillary data sets.

"""
from logging import Logger
from typing import Dict, Set
import re
import xml.etree.ElementTree as ET
import yaml

from harmony.util import Config

from pymods.cf_config import CFConfig
from pymods.utilities import (DAP4_TO_NUMPY_MAP, download_url,
                              get_xml_namespace, split_attribute_path,
                              recursive_get)
from pymods.variable import Variable


class VarInfo:
    """ A class to represent the full dataset of a granule, having downloaded
        and parsed a `.dmr` file from OPeNDAP.

    """

    def __init__(self, file_url: str, logger: Logger, temp_dir: str,
                 access_token: str, env_config: Config,
                 config_file: str = 'pymods/var_subsetter_config.yml'):
        """ Distinguish between variables containing references to other
            datasets, and those that do not. The former are considered science
            variables, providing they are not considered coordinates or
            dimensions for another variable.

            Unlike NCInfo in SwotRepr, each variable contains references to
            their specific coordinates and dimensions, allowing the retrieval
            of all required variables for a specified list of science
            variables.

        """
        self.config_file = config_file
        self.logger = logger
        self.output_dir = temp_dir
        self.cf_config = None
        self.global_attributes = {}
        self.short_name = None
        self.mission = None
        self.namespace = None
        self.metadata_variables: Dict[str, Variable] = {}
        self.variables_with_coordinates: Dict[str, Variable] = {}
        self.references: Set[str] = set()
        self.metadata = {}

        self._set_var_info_config()
        self._read_dataset(file_url, access_token, env_config)
        self._set_global_attributes()
        self._set_mission_and_short_name()
        self._set_cf_config()
        self._update_global_attributes()
        self._extract_variables()

    def _read_dataset(self, dataset_url: str, access_token: str, config: Config):
        """ Download a `.dmr` file from the specified URL. Then extract the XML
            tree and namespace.

        """
        self.logger.info('Retrieving .dmr from OPeNDAP')
        dmr_file = download_url(dataset_url, self.output_dir, self.logger, access_token, config)

        with open(dmr_file, 'r') as file_handler:
            dmr_content = file_handler.read()

        self.dataset = ET.fromstring(dmr_content)
        self.namespace = get_xml_namespace(self.dataset)

    def _set_global_attributes(self):
        """ Recurse through all attributes in a `.dmr` file. Starting at the
            supplied root element, find all child Attribute elements. Those
            children with a type property corresponding to a DAP4 variable
            type are placed in an output_dictionary. If the type is not
            recognised by the DAP4 protocol, the attribute is assumed to be a
            string.

        """
        def save_attribute(output, group_path, attribute):
            attribute_name = attribute.get('name')
            attribute_value = attribute.find(f'{self.namespace}Value').text
            dap4_type = attribute.get('type')
            numpy_type = DAP4_TO_NUMPY_MAP.get(dap4_type, str)

            group_dictionary = output

            if group_path != '':
                # Recurse through group keys to retrieve the nested group to
                # which the attribute belongs. If a group in the path doesn't
                # exist, because this attribute is the first to be parsed from
                # this group, then create a new nested dictionary for the group
                # to contain the child attributes
                nested_groups = group_path.lstrip('/').split('/')
                for group in nested_groups:
                    group_dictionary = group_dictionary.setdefault(group, {})

            group_dictionary[attribute_name] = numpy_type(attribute_value)

        self.traverse_elements(self.dataset, {'Attribute'}, save_attribute,
                               self.global_attributes)

    def _extract_variables(self):
        """ Iterate through all children of the `.dmr` root dataset element.
            If the child matches one of the DAP4 variable types, then create an
            instance of the `Variable` class, and assign it to either the
            `variables_with_coordinates` or the `metadata_variables`
            dictionary accordingly.

        """
        def save_variable(output, group_path, element):
            element_path = '/'.join([group_path, element.get('name')])
            variable = Variable(element, self.cf_config,
                                namespace=self.namespace,
                                full_name_path=element_path)
            output[variable.full_name_path] = variable
            self._assign_variable(variable)

        all_variables = {}

        self.traverse_elements(self.dataset, set(DAP4_TO_NUMPY_MAP.keys()),
                               save_variable, all_variables)

    def _assign_variable(self, variable_object):
        """ Given a `Variable` instance, based on the content of the
            `coordinates` attribute, assign it to either the
            `variables_with_coordinates` or `metadata_variables` dictionary.
            Additionally, the set of references for all variables is updated.

        """
        full_path = variable_object.full_name_path
        self.references.update(variable_object.get_references())

        if variable_object.coordinates:
            self.variables_with_coordinates[full_path] = variable_object
        else:
            self.metadata_variables[full_path] = variable_object

    def _set_var_info_config(self):
        """ Read the VarInfo configuration YAML file, containing locations to
            search for the collection short_name attribute, and the mapping
            from short_name to satellite mission.

        """
        with open(self.config_file, 'r') as file_handler:
            self.var_info_config = yaml.load(file_handler, yaml.FullLoader)

    def _set_cf_config(self):
        """ Instantiate a CFConfig object, to contain any rules for exclusions,
            required fields and augmentations to CF attributes that are not
            contained within a granule from the specified collection.

        """
        self.cf_config = CFConfig(self.mission, self.short_name,
                                  self.config_file)

    def _set_mission_and_short_name(self):
        """ Check a series of potential locations for the collection short name
        of the granule. Once that is determined, match that short name to its
        associated mission.

        """
        self.short_name = next(
            (recursive_get(self.global_attributes, split_attribute_path(item))
             for item
             in self.var_info_config['Collection_ShortName_Path']
             if recursive_get(self.global_attributes,
                              split_attribute_path(item))
             is not None),
            None
        )

        if self.short_name is not None:
            self.mission = next((name
                                 for pattern, name
                                 in self.var_info_config['Mission'].items()
                                 if re.match(pattern, self.short_name)
                                 is not None), None)

    def _update_global_attributes(self):
        """ Having identified the mission and short_name for the granule, and
            therefore obtained the relevant CF configuration overrides and
            supplements, update the global attributes for this granule using
            the CFConfig class instance. As the overrides are assumed to have
            the strongest priority, the dictionary is updated with these values
            last.

        """
        if self.cf_config.global_supplements:
            self.global_attributes.update(self.cf_config.global_supplements)

        if self.cf_config.global_overrides:
            self.global_attributes.update(self.cf_config.global_overrides)

    def get_science_variables(self) -> Set[str]:
        """ Retrieve set of names for all variables that have coordinate
            references, that are not themselves used as dimensions, coordinates
            or ancillary date for another variable.

        """
        exclusions_pattern = re.compile(
            '|'.join(self.cf_config.excluded_science_variables)
        )

        filtered_with_coordinates = {
            variable
            for variable
            in self.variables_with_coordinates
            if variable is not None
            and not self.variable_is_excluded(variable, exclusions_pattern)
        }

        return filtered_with_coordinates - self.references

    def get_metadata_variables(self) -> Set[str]:
        """ Retrieve set of names for all variables that do no have
            coordaintes references, that are not themselves used as dimensions,
            coordinates or ancillary data for another variable.

            Additionally, any excluded science variables, that are contained
            in the variables_with_coordinates class attribute should be
            considered a metadata variable.

        """
        exclusions_pattern = re.compile(
            '|'.join(self.cf_config.excluded_science_variables)
        )

        additional_metadata = {
            variable
            for variable
            in self.variables_with_coordinates
            if variable is not None
            and self.variable_is_excluded(variable, exclusions_pattern)
        }

        metadata_variables = set(self.metadata_variables.keys())
        metadata_variables.update(additional_metadata)

        return metadata_variables - self.references

    @staticmethod
    def variable_is_excluded(variable_name: str,
                             exclusions_pattern: re.Pattern) -> bool:
        """ Ensure the variable name does not match any collection specific
            exclusion rules.

        """
        if exclusions_pattern.pattern != '':
            exclude_variable = exclusions_pattern.match(variable_name) is not None
        else:
            exclude_variable = False

        return exclude_variable

    def get_required_variables(self, requested_variables: Set[str]) -> Set[str]:
        """ Retrieve requested variables and recursively search for all
            associated dimension and coordinate variables. The returned set
            should be the union of the science variables, coordinates and
            dimensions.

            The requested variables are also augmented to include required
            variables for the collection, as indicated by the CFConfig class
            instance, and any references within those variables.

        """
        if self.cf_config.required_variables:
            cf_required_pattern = re.compile(
                '|'.join(self.cf_config.required_variables)
            )

            all_variable_names = set(self.variables_with_coordinates.keys()).union(
                set(self.metadata_variables.keys())
            )

            cf_required_variables = {variable
                                     for variable
                                     in all_variable_names
                                     if variable is not None
                                     and re.match(cf_required_pattern, variable)}
        else:
            cf_required_variables = set()

        requested_variables.update(cf_required_variables)
        required_variables: Set[str] = set()

        while len(requested_variables) > 0:
            variable_name = requested_variables.pop()
            required_variables.add(variable_name)

            variable = (self.variables_with_coordinates.get(variable_name) or
                        self.metadata_variables.get(variable_name))

            if variable is not None:
                # Add variable. Enqueue references not already present in
                # required set.
                variable_references = variable.get_references()
                requested_variables.update(
                    variable_references.difference(required_variables)
                )

        return self.exclude_fake_dimensions(required_variables)

    @staticmethod
    def exclude_fake_dimensions(variable_set: Set[str]) -> Set[str]:
        """ An OPeNDAP `.dmr` can contain fake dimensions, used to supplement
            missing information for a granule. These cannot be retrieved when
            requesting a subset from an OPeNDAP server, and must be removed
            from the list of required variables.

        """
        fakedim_pattern = re.compile(r'.*/FakeDim\d+')

        return {variable for variable in variable_set
                if not fakedim_pattern.match(variable)}

    def traverse_elements(self, element: ET.Element, element_types: Set[str],
                          operation, output, group_path: str = ''):
        """ Perform a depth first search of the `.dmr` `Dataset` element.
            When a variable is located perform an operation on the supplied
            output object, using the supplied function or class.

        """

        for child in list(element):
            # If it is in the DAP4 list: use the function
            # else, if it is a Group, call this function again
            element_type = child.tag.replace(self.namespace, '')

            if element_type in element_types:
                operation(output, group_path, child)
            elif element_type == 'Group':
                new_group_path = '/'.join([group_path, child.get('name')])
                self.traverse_elements(child, element_types, operation, output,
                                       new_group_path)
