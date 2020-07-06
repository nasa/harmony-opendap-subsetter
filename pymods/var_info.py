""" This module contains a class designed to read information from a dmr
    file. This should group the input into science variables, metadata,
    coordinates, dimensions and ancillary data sets.

"""
from logging import Logger
from typing import Dict, List, Optional, Set, Tuple, Union
import os
import re
import xml.etree.ElementTree as ET
import yaml

from pydap.client import open_url
from pydap.model import BaseType, DatasetType
from webob.exc import HTTPError, HTTPRedirection
import requests

from pymods.cf_config import CFConfig
from pymods.exceptions import PydapRetrievalError
from pymods.utilities import (DAP4_TO_NUMPY_MAP, get_xml_attribute,
                              get_xml_namespace, pydap_attribute_path,
                              recursive_get)


VariableType = Union[BaseType, ET.Element]


class VarInfo:
    """ A class to represent the full dataset of a granule, having read
        information from the dmr or dmrpp for it.

    """

    def __init__(self, file_url: str, logger: Logger,
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
        self.cf_config = None
        self.global_attributes = None
        self.short_name = None
        self.mission = None
        self.namespace = None
        self.metadata_variables: Dict[str, Variable] = {}
        self.variables_with_coordinates: Dict[str, Variable] = {}
        self.references: Set[str] = set()
        self.metadata = {}
        self.name_map: Dict[str, str] = dict()

        self._set_var_info_config()

        if file_url.endswith('.dmr'):
            self._read_dataset_from_dmr(file_url)
        else:
            self._read_dataset_from_pydap(file_url)

        self._set_name_map()
        self._set_global_attributes()
        self._set_mission_and_short_name()
        self._set_cf_config()
        self._update_global_attributes()

        if isinstance(self.dataset, DatasetType):
            self._extract_pydap_variables()
        else:
            self._extract_dmr_variables()

    def _read_dataset_from_dmr(self, dmr_url: str):
        """ This method parses the downloaded dmr file at the specified URL.
            For each variable in the dataset, an object is created, while
            resolving any associated references, and updating those sets using
            the CFConfig class to account for known data issues.

        """
        self.logger.info('Retrieving .dmr from OPeNDAP')
        # TODO: Replace with work from DAS-700 (a common function in
        # pymods/utilities.py)
        user = os.environ.get('EDL_USERNAME')
        password = os.environ.get('EDL_PASSWORD')
        dmr_response = requests.get(dmr_url, auth=(user, password))

        self.dataset = ET.fromstring(dmr_response.content)
        self.namespace = get_xml_namespace(self.dataset)

    def _read_dataset_from_pydap(self, pydap_url: str):
        """ This method parses the downloaded pydap dataset at the specified
            URL. For each variable in the dataset, an object is created, while
            resolving any associated references, and updating those sets using
            the CFConfig class to account for known data issues.

        """
        self.logger.info('Retrieving dataset from pydap')
        try:
            self.dataset = open_url(pydap_url)
        except (HTTPError, HTTPRedirection) as error:
            self.logger.error(f'{error.status_code} Error retrieving pydap '
                              'dataset')
            raise PydapRetrievalError(error.comment)

    def _set_name_map(self):
        """ Create a mapping from pydap or XML variable name to the full path.
            (e.g.: "group_variable" to "/group/variable").

        """
        if isinstance(self.dataset, DatasetType):
            self.name_map = {
                pydap_name: variable.attributes.get('fullnamepath', pydap_name)
                for pydap_name, variable
                in self.dataset.items()
            }
        else:
            self.name_map = {
                f'/{child.get("name")}': get_xml_attribute(
                    child, 'fullnamepath', self.namespace, child.get('name')
                )
                for child in self.dataset
                if child.tag.lstrip(self.namespace) in DAP4_TO_NUMPY_MAP
            }

    def _extract_pydap_variables(self):
        """ Iterate through all the variables returned in a `pydap` Dataset
            instance. For each variable, create an instance of the `Variable`
            class, and assign it to either the `variables_with_coordinates` or
            the `metadata_variables` dictionary accordingly.

        """
        for variable in self.dataset.values():
            variable_object = Variable(variable, self.cf_config, self.name_map)
            self._assign_variable(variable_object)

    def _extract_dmr_variables(self):
        """ Iterate through all child tags of the `.dmr` root dataset element.
            If the tag matches one of the DAP4 variable types, then create an
            instance of the `Variable` class, and assign it to either the
            `variables_with_coordinates` or the `metadata_variables`
            dictionary accordingly.

        """
        for child in self.dataset:
            if child.tag.lstrip(self.namespace) in DAP4_TO_NUMPY_MAP:
                variable_object = Variable(child, self.cf_config,
                                           self.name_map, self.namespace)
                self._assign_variable(variable_object)

    def _assign_variable(self, variable_object):
        """ A function combining the common operations once a variable from
            either a `pydap` Dataset or `.dmr` file has been identified.

            Given a `Variable` instance, based on the content of the
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
            (recursive_get(self.global_attributes, pydap_attribute_path(item))
             for item
             in self.var_info_config['Collection_ShortName_Path']
             if recursive_get(self.global_attributes,
                              pydap_attribute_path(item))
             is not None),
            None
        )

        if self.short_name is not None:
            self.mission = next((name
                                 for pattern, name
                                 in self.var_info_config['Mission'].items()
                                 if re.match(pattern, self.short_name)
                                 is not None), None)

    def _set_global_attributes(self):
        """ Check the attributes of the returned pydap dataset for both the
            HDF5_GLOBAL and NC_GLOBAL keys. Use whichever of these has any
            valid keys within it. If neither have keys, then return an empty
            dictionary.

        """
        if isinstance(self.dataset, DatasetType):
            self.global_attributes = self.dataset.attributes
        else:
            self.global_attributes = {}
            self._extract_attributes_from_dmr(self.dataset,
                                              self.global_attributes)

    def _extract_attributes_from_dmr(self, root_element: ET.Element,
                                     output_attributes: Dict):
        """ Recurse through all attributes in a `.dmr` file. Starting at the
            supplied root element, find all child Attribute elements. Those
            children with a type property corresponding to a DAP4 variable
            type are placed in an output_dictionary. If the Attribute tag has a
            `Container` type, containing children of it's own, parse its
            children in the same way.

        """
        for attribute in root_element.findall(f'{self.namespace}Attribute'):
            attribute_name = attribute.get('name')
            attribute_type = attribute.get('type')
            if attribute_type in DAP4_TO_NUMPY_MAP:
                value_element = attribute.find(f'{self.namespace}Value')
                output_attributes[attribute_name] = (
                    DAP4_TO_NUMPY_MAP[attribute_type](value_element.text)
                )
            elif attribute_type == 'Container':
                output_attributes[attribute_name] = {}
                self._extract_attributes_from_dmr(
                    attribute, output_attributes[attribute_name]
                )

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
            and not re.match(exclusions_pattern, variable)
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

        additional_metadata = {variable
                               for variable
                               in self.variables_with_coordinates
                               if variable is not None
                               and re.match(exclusions_pattern, variable)}

        metadata_variables = set(self.metadata_variables.keys())
        metadata_variables.update(additional_metadata)

        return metadata_variables - self.references

    def get_required_variables(self, requested_variables: Set[str]) -> Set[str]:
        """ Retrieve requested variables and recursively search for all
            associated dimension and coordinate variables. The returned set
            should be the union of the science variables, coordinates and
            dimensions.

            The requested variables are also augmented to include required
            variables for the collection, as indicated by the CFConfig class
            instance, and any references within those variables.

        """
        # TODO: Assess performance of recursive reference search including
        # CFConfig defined required fields (which could be in the hundreds).
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

        return required_variables


class Variable:
    """ A class to represent a single variable contained within a granule.
        This class maps from either the `pydap.model.BaseType` class, or an
        XML element from a dmr file, to a format specifically for `VarInfo`,
        in which references are fully qualified, and also augmented by any
        overrides or supplements from the variable subsetter configuration
        file.

    """

    def __init__(self, variable: VariableType, cf_config: CFConfig,
                 name_map: Dict[str, str], namespace: Optional[str] = None):
        """ Extract the references contained within the variable's coordinates,
            ancillary_variables or dimensions. These should be augmented by
            information from the CFConfig instance passed to the class.

            Additionally, store other information required for UMM-Var record
            production in an attributes dictionary. (These attributes may not
            be an exhaustive list).

        """
        self.namespace = namespace
        self.full_name_path = self._get_attribute(variable, 'fullnamepath')
        self.cf_config = cf_config.get_cf_attributes(self.full_name_path)
        self.group_path, self.name = self._extract_group_and_name(variable)

        self.ancillary_variables = self._get_cf_references(variable,
                                                           'ancillary_variables')
        self.coordinates = self._get_cf_references(variable, 'coordinates')
        self.subset_control_variables = self._get_cf_references(
            variable, 'subset_control_variables'
        )
        self.dimensions = self._extract_dimensions(variable, name_map)

        self.attributes = {
            'acquisition_source_name': self._get_attribute(variable, 'source'),
            'data_type': self._get_data_type(variable),
            'definition': self._get_attribute(variable, 'description'),
            'fill_value': self._get_attribute(variable, '_FillValue'),
            'long_name': self._get_attribute(variable, 'long_name'),
            'offset': self._get_attribute(variable, 'offset', 0),
            'scale': self._get_attribute(variable, 'scale', 1),
            'units': self._get_attribute(variable, 'units'),
            'valid_max': self._get_attribute(variable, 'valid_max'),
            'valid_min': self._get_attribute(variable, 'valid_min')
        }

    def get_references(self) -> Set[str]:
        """ Combine the references extracted from the ancillary_variables,
            coordinates and dimensions data into a single set for VarInfo to
            use directly.

        """
        return self.ancillary_variables.union(self.coordinates,
                                              self.dimensions,
                                              self.subset_control_variables)

    def _get_data_type(self, variable: VariableType) -> str:
        """ Extract a string representation of the variable data type. """
        if isinstance(variable, BaseType):
            data_type = variable.dtype.name
        else:
            data_type = variable.tag.lstrip(self.namespace).lower()

        return data_type

    def _get_attribute(self, variable: VariableType, attribute_name: str,
                       default_value: Optional = None) -> Optional:
        """ Determine if the variable is from pydap or is XML. Then use the
            appropriate functionality to extract the attribute value, falling
            back to a default value if the attribute is absent.

        """
        if isinstance(variable, BaseType):
            attribute = variable.attributes.get(attribute_name, default_value)
        else:
            attribute = get_xml_attribute(variable, attribute_name,
                                          self.namespace, default_value)

        return attribute

    def _get_cf_references(self, variable: VariableType,
                           attribute_name: str) -> Set[str]:
        """ Obtain the string value of a metadata attribute, which should have
            already been corrected for any known artefacts (missing or
            incorrect references). Then split this string and qualify the
            individual references.

        """
        attribute_string = self._get_cf_attribute(variable, attribute_name)
        return self._extract_references(attribute_string)

    def _get_cf_attribute(self, variable: VariableType,
                          attribute_name: str) -> str:
        """ Given the name of a CF-convention attribute, extract the string
            value from the variable metadata. Then check the output from the
            CF configuration file, to see if this value should be replaced, or
            supplemented with more data.

        """
        cf_overrides = self.cf_config['cf_overrides'].get(attribute_name)
        cf_supplements = self.cf_config['cf_supplements'].get(attribute_name)

        if cf_overrides is not None:
            attribute_value = cf_overrides
        else:
            attribute_value = self._get_attribute(variable, attribute_name)
        if cf_supplements is not None and attribute_value is not None:
            attribute_value += f', {cf_supplements}'
        elif cf_supplements is not None:
            attribute_value = cf_supplements

        return attribute_value

    def _extract_references(self, attribute_string: str) -> Set[str]:
        """ Given a string value of an attribute, which may contain multiple
            references to dataset, split that string based on either commas,
            or spaces (or both together). Then if any reference is a relative
            path, make it absolute.

        """
        if attribute_string is not None:
            raw_references = re.split(r'\s+|,\s*', attribute_string)
            references = self._qualify_references(raw_references)
        else:
            references = set()

        return references

    def _extract_dimensions(self, variable: VariableType,
                            name_map: Dict[str, str]) -> Set[str]:
        """ Find the dimensions for the variable in question. If there are
            overriding or supplemental dimensions from the CF configuration
            file, these are used instead of, or in addtion to, the raw
            dimensions from pydap. All references are converted to absolute
            paths in the granule. A set of all fully qualified references is
            returned.

            The dimensions stored in a pydap BaseType object or dmr XML
            variable have underscores in place of slashes, so the `name_map`
            tries to find the original full path for the dimension variables.
            If the dimension is not found in the `name_map`, the original
            string is used.

        """
        overrides = self.cf_config['cf_overrides'].get('dimensions')
        supplements = self.cf_config['cf_supplements'].get('dimensions')

        if overrides is not None:
            dimensions = re.split(r'\s+|,\s*', overrides)
        else:
            if isinstance(variable, BaseType):
                raw_dimensions = variable.dimensions
            else:
                raw_dimensions = [dimension.get('name')
                                  for dimension
                                  in variable.findall(f'{self.namespace}Dim')]

            dimensions = [name_map.get(dimension, dimension)
                          for dimension in raw_dimensions
                          if dimension is not None]

        if supplements is not None:
            dimensions += re.split(r'\s+|,\s*', supplements)

        return self._qualify_references(dimensions)

    def _qualify_references(self, raw_references: List[str]) -> Set[str]:
        """ Take a list of local references to other variables, and produce a
            list of absolute references.

        """
        if self.group_path is not None:
            references = set()
            for reference in raw_references:
                if reference.startswith('../'):
                    # Reference is relative, and requires manipulation
                    absolute_path = self._construct_absolute_path(reference)
                elif reference.startswith('/'):
                    # Reference is already absolute
                    absolute_path = reference
                elif reference.startswith('./'):
                    # Reference is in the same group as this variable
                    absolute_path = self.group_path + reference[1:]
                else:
                    # Reference is in the same group as this variable
                    absolute_path = '/'.join([self.group_path, reference])

                references.add(absolute_path)

        else:
            references = set(raw_references)

        return references

    def _construct_absolute_path(self, reference: str) -> str:
        """ For a relative reference to another variable (e.g. '../latitude'),
            construct an absolute path by combining the reference with the
            group path of the variable.

        """
        relative_prefix = '../'
        group_path_pieces = self.group_path.split('/')

        while reference.startswith(relative_prefix):
            reference = reference[len(relative_prefix):]
            group_path_pieces.pop()

        absolute_path = group_path_pieces + [reference]
        return '/'.join(absolute_path)

    def _extract_group_and_name(self, variable: BaseType) -> Tuple[str]:
        """ Check if the 'fullpathname' attribute is defined. If so, derive the
            group and local name of the variable.

        """
        if self.full_name_path is not None:
            split_full_path = self.full_name_path.split('/')
            name = split_full_path.pop(-1)
            group_path = '/'.join(split_full_path) or None
        else:
            if isinstance(variable, BaseType):
                name = variable.name
            else:
                name = variable.get('name')

            group_path = None

        return group_path, name
