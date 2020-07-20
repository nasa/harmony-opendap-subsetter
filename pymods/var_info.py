""" This module contains a class designed to read information from a dmr
    file. This should group the input into science variables, metadata,
    coordinates, dimensions and ancillary data sets.

"""
from abc import ABC, abstractmethod
from logging import Logger
from typing import Dict, Set, Union
import re
import xml.etree.ElementTree as ET
import yaml

from harmony.util import download as util_download
from pydap.client import open_url
from webob.exc import HTTPError, HTTPRedirection

from pymods.cf_config import CFConfig
from pymods.exceptions import PydapRetrievalError
from pymods.utilities import (DAP4_TO_NUMPY_MAP, get_xml_attribute,
                              get_xml_namespace, pydap_attribute_path,
                              recursive_get)
from pymods.variable import VariableFromDmr, VariableFromPydap


OutputVariableType = Union[VariableFromDmr, VariableFromPydap]


class VarInfoBase(ABC):
    """ An abstract base class to represent the full dataset of a granule,
        having read information from either a `pydap` `Dataset` or downloading
        a `.dmr` from OPeNDAP.

    """

    def __init__(self, file_url: str, logger: Logger, temp_dir: str,
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
        self.metadata_variables: Dict[str, OutputVariableType] = {}
        self.variables_with_coordinates: Dict[str, OutputVariableType] = {}
        self.references: Set[str] = set()
        self.metadata = {}
        self.name_map: Dict[str, str] = dict()

        self._set_var_info_config()
        self._read_dataset(file_url)
        self._set_name_map()
        self._set_global_attributes()
        self._set_mission_and_short_name()
        self._set_cf_config()
        self._update_global_attributes()
        self._extract_variables()

    @abstractmethod
    def _read_dataset(self, dataset_url: str):
        """ This method parses the downloaded dmr file at the specified URL.
            For each variable in the dataset, an object is created, while
            resolving any associated references, and updating those sets using
            the CFConfig class to account for known data issues.

        """

    @abstractmethod
    def _set_name_map(self):
        """ Create a mapping from a variable name to the full path for that
            variable (e.g.: "group_variable" to "/group/variable").

        """

    @abstractmethod
    def _set_global_attributes(self):
        """ Check the attributes of the returned `pydap` `Dataset` or `.dmr`
            XML tree.

        """

    @abstractmethod
    def _extract_variables(self):
        """ Iterate through all the variables in the retrieved dataset. For
            each variable, create an instance of a `VariableFromDmr ` or
            `VariableFromPydap` class, and assign it to either the
            `variables_with_coordinates` or the `metadata_variables` dictionary
            accordingly.

        """

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


class VarInfoFromDmr(VarInfoBase):
    """ A child class that inherits from `VarInfoBase` and implements functions
        to retrieve a dataset from a `.dmr` file, and then extract variables
        from the resulting XML tree.

    """
    def _read_dataset(self, dataset_url: str):
        """ Download a `.dmr` file from the specified URL. Then extract the XML
            tree and namespace.

        """
        self.logger.info('Retrieving .dmr from OPeNDAP')
        dmr_file = util_download(dataset_url, self.output_dir, self.logger)

        with open(dmr_file, 'r') as file_handler:
            dmr_content = file_handler.read()

        self.dataset = ET.fromstring(dmr_content)
        self.namespace = get_xml_namespace(self.dataset)

    def _set_name_map(self):
        """ Create a mapping from an XML variable name to the full path.
            (e.g.: "group_variable" to "/group/variable").

        """
        self.name_map = {
            f'/{child.get("name")}': get_xml_attribute(
                child, 'fullnamepath', self.namespace, child.get('name')
            )
            for child in self.dataset
            if child.tag.lstrip(self.namespace) in DAP4_TO_NUMPY_MAP
        }

    def _set_global_attributes(self):
        """ Check the attributes of the returned `.dmr` XML tree. """
        self._extract_attributes_from_dmr(self.dataset, self.global_attributes)

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

    def _extract_variables(self):
        """ Iterate through all child tags of the `.dmr` root dataset element.
            If the tag matches one of the DAP4 variable types, then create an
            instance of the `Variable` class, and assign it to either the
            `variables_with_coordinates` or the `metadata_variables`
            dictionary accordingly.

        """
        for child in self.dataset:
            if child.tag.lstrip(self.namespace) in DAP4_TO_NUMPY_MAP:
                variable_object = VariableFromDmr(child, self.cf_config,
                                                  self.name_map,
                                                  self.namespace)
                self._assign_variable(variable_object)


class VarInfoFromPydap(VarInfoBase):
    """ A child class that inherits from `VarInfoBase` and implements functions
        to retrieve a dataset from `pydap`, and then extract variables from the
        resulting `Dataset` object..

    """
    def _read_dataset(self, dataset_url: str):
        """ Download a `pydap` `Dataset` from the specified URL. """
        self.logger.info('Retrieving dataset from pydap')
        try:
            self.dataset = open_url(dataset_url)
        except (HTTPError, HTTPRedirection) as error:
            self.logger.error(f'{error.status_code} Error retrieving pydap '
                              'dataset')
            raise PydapRetrievalError(error.comment)

    def _set_name_map(self):
        """ Create a mapping from pydap variable name to the full path.
            (e.g.: "group_variable" to "/group/variable").

        """
        self.name_map = {
            pydap_name: variable.attributes.get('fullnamepath', pydap_name)
            for pydap_name, variable
            in self.dataset.items()
        }

    def _set_global_attributes(self):
        """ Check the attributes of the returned `pydap` `Dataset`. """
        self.global_attributes = self.dataset.attributes

    def _extract_variables(self):
        """ Iterate through all the variables returned in a `pydap` `Dataset`
            instance. For each variable, create an instance of the
            `VariableFromPydap` class, and assign it to either the
            `variables_with_coordinates` or the `metadata_variables` dictionary
            accordingly.

        """
        for variable in self.dataset.values():
            variable_object = VariableFromPydap(variable, self.cf_config,
                                                self.name_map)
            self._assign_variable(variable_object)
