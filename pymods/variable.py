""" This module contains a class designed to read information from a dmr
    file. This should group the input into science variables, metadata,
    coordinates, dimensions and ancillary data sets.

"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union
import re
import xml.etree.ElementTree as ET

from pydap.model import BaseType

from pymods.cf_config import CFConfig
from pymods.utilities import get_xml_attribute


InputVariableType = Union[BaseType, ET.Element]


class VariableBase(ABC):
    """ A class to represent a single variable contained within a granule.
        This parent class contains the common functionality for a variable
        defined by either the `pydap.model.BaseType` class, or an XML element
        from a `.dmr` file. It will produce a format specifically for
        `VarInfoFromDmr` and `VarInfoFromPydap`, in which references are fully
        qualified, and also augmented by any overrides or supplements from the
        variable subsetter configuration file.

    """

    def __init__(self, variable: BaseType, cf_config: CFConfig,
                 name_map: Dict[str, str], namespace: Optional[str] = None,
                 full_name_path: Optional[str] = None):
        """ Extract the references contained within the variable's coordinates,
            ancillary_variables or dimensions. These should be augmented by
            information from the CFConfig instance passed to the class.

            Additionally, store other information required for UMM-Var record
            production in an attributes dictionary. (These attributes may not
            be an exhaustive list).

        """
        self.namespace = namespace

        if full_name_path is not None:
            self.full_name_path = full_name_path
        else:
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

    @abstractmethod
    def _get_variable_name(self, variable: InputVariableType):
        """ Extract the name of the variable. """

    @abstractmethod
    def _get_data_type(self, variable: InputVariableType) -> Optional[str]:
        """ Extract a string representation of the variable data type. """

    @abstractmethod
    def _get_attribute(self, variable: InputVariableType, attribute_name: str,
                       default_value: Optional = None) -> Optional:
        """ Extract the attribute value, falling back to a default value if the
            attribute is absent.

        """

    @staticmethod
    @abstractmethod
    def _get_raw_dimensions(variable: InputVariableType):
        """ Retrieve the dimension names as they are stored within the
            variable.

        """

    def get_references(self) -> Set[str]:
        """ Combine the references extracted from the ancillary_variables,
            coordinates and dimensions data into a single set for VarInfo to
            use directly.

        """
        return self.ancillary_variables.union(self.coordinates,
                                              self.dimensions,
                                              self.subset_control_variables)

    def _get_cf_references(self, variable: InputVariableType,
                           attribute_name: str) -> Set[str]:
        """ Obtain the string value of a metadata attribute, which should have
            already been corrected for any known artefacts (missing or
            incorrect references). Then split this string and qualify the
            individual references.

        """
        attribute_string = self._get_cf_attribute(variable, attribute_name)
        return self._extract_references(attribute_string)

    def _get_cf_attribute(self, variable: InputVariableType,
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

    def _extract_dimensions(self, variable: InputVariableType,
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
        inverse_mapping = {with_underscores: with_slashes
                           for with_slashes, with_underscores
                           in name_map.items()}

        if overrides is not None:
            dimensions = re.split(r'\s+|,\s*', overrides)
        else:
            dimensions = [inverse_mapping.get(dimension, dimension)
                          for dimension in self._get_raw_dimensions(variable)
                          if dimension is not None]

        if supplements is not None:
            dimensions += re.split(r'\s+|,\s*', supplements)

        return self._qualify_references(dimensions)

    def _qualify_references(self, raw_references: List[str]) -> Set[str]:
        """ Take a list of local references to other variables, and produce a
            list of absolute references.

        """
        references = set()

        if self.group_path is not None:
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
            for reference in raw_references:
                if reference.startswith('/'):
                    absolute_path = reference
                else:
                    absolute_path = f'/{reference}'

                references.add(absolute_path)

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


class VariableFromDmr(VariableBase):
    """ This child class inherits from the `VariableBase` class, and implements
        the abstract methods assuming the variable source is part of an XML
        element tree.

    """
    def _get_data_type(self, variable: ET.Element) -> str:
        """ Extract a string representation of the variable data type. """
        return variable.tag.lstrip(self.namespace).lower()

    def _get_attribute(self, variable: ET.Element, attribute_name: str,
                       default_value: Optional = None) -> Optional:
        """ Use a utility function to retrieve an attribute from a Variable
            XML tag in the `.dmr`. If the attribute is absent, use the
            provided default value.

        """
        return get_xml_attribute(variable, attribute_name, self.namespace,
                                 default_value)

    def _get_raw_dimensions(self, variable: ET.Element) -> List[str]:
        """ Extract the raw dimension names from a Dim XML tag. """
        return [dimension.get('name')
                for dimension
                in variable.findall(f'{self.namespace}Dim')]

    @staticmethod
    def _get_variable_name(variable: ET.Element) -> str:
        """ Extract the raw variable name from a Variable XML tag. """
        return variable.get('name')


class VariableFromPydap(VariableBase):
    """ This child class inherits from the `VariableBase` class, and implements
        the abstract methods assuming the variable source is a `pydap
        ` variable.

    """
    def _get_data_type(self, variable: BaseType) -> str:
        """ Extract the string representation of the `pydap` variable. """
        return variable.dtype.name

    def _get_attribute(self, variable: BaseType, attribute_name: str,
                       default_value: Optional = None) -> Optional:
        """ Retreive the value of an attribute from the `pydap` `BaseType`
            dictionary of attributes. If the attribute is absent and a default
            value is given, return that default. """
        return variable.attributes.get(attribute_name, default_value)

    def _get_raw_dimensions(self, variable: BaseType) -> List[str]:
        """ Extract the raw strings of all dimensions associated with the
            `BaseType` variable.

        """
        return variable.dimensions

    @staticmethod
    def _get_variable_name(variable: BaseType) -> str:
        """ Extract the raw variable name from the `pydap` `BaseType`. """
        return variable.name
