""" This module contains a class designed to read information from a dmr
    file. This should group the input into science variables, metadata,
    coordinates, dimensions and ancillary data sets.

"""
from logging import Logger
from typing import Dict, List, Set, Tuple
import re
import yaml

from pydap.client import open_url
from pydap.model import BaseType
from webob.exc import HTTPError, HTTPRedirection

from pymods.cf_config import CFConfig
from pymods.exceptions import PydapRetrievalError
from pymods.utilities import pydap_attribute_path, recursive_get


class VarInfo:
    """ A class to represent the full dataset of a granule, having read
        information from the dmr or dmrpp for it.

    """

    def __init__(self, pydap_url: str, logger: Logger,
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
        self.metadata_variables: Dict[str, Variable] = {}
        self.variables_with_coordinates: Dict[str, Variable] = {}
        self.references: Set[str] = set()
        self.metadata = {}

        self._set_var_info_config()
        self._read_dataset_from_pydap(pydap_url)

    def _read_dataset_from_pydap(self, file_url: str):
        """ This method parses the downloaded pydap dataset at the specified
            URL. For each variable in the dataset, an object is created, while
            resolving and associated references, and updating those sets using
            the CFConfig class to account for known data issues.

        """
        try:
            self.pydap_dataset = open_url(file_url)
        except (HTTPError, HTTPRedirection) as error:
            self.logger.error(f'{error.status_code} Error retrieving pydap '
                              'dataset')
            raise PydapRetrievalError(error.comment)

        self._set_global_attributes()
        self._set_mission_and_short_name()
        self._set_cf_config()
        self._update_global_attributes()

        for variable in self.pydap_dataset.values():
            # TODO: When receiving a non-flattened file, augment this to make
            # sure it is a BaseType object, if StructureType or SequenceType
            # make sure these are correctly handled.
            variable_object = Variable(variable, self.cf_config)
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
        hdf5_global_attributes = self.pydap_dataset.attributes.get('HDF5_GLOBAL')
        nc_global_attributes = self.pydap_dataset.attributes.get('NC_GLOBAL')

        if bool(hdf5_global_attributes):
            self.global_attributes = hdf5_global_attributes
        elif bool(nc_global_attributes):
            self.global_attributes = nc_global_attributes
        else:
            self.global_attributes = {}

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

        requested_variables.update(cf_required_variables)
        required_variables: Set[str] = set()

        while len(requested_variables) > 0:
            variable_name = requested_variables.pop()
            variable = (self.variables_with_coordinates.get(variable_name) or
                        self.metadata_variables.get(variable_name))

            if variable is not None:
                # Add variable. Enqueue coordinates and dimensions not already
                # present in required set.
                required_variables.add(variable_name)
                requested_variables.update(
                    variable.coordinates.difference(required_variables)
                )
                requested_variables.update(
                    variable.dimensions.difference(required_variables)
                )

        return required_variables


class Variable:
    """ A class to represent a single variable contained within a granule.
        This class maps from the `pydap.model.BaseType` class to a format
        specifically for `VarInfo`, in which references are fully qualified,
        and also augmented by any overrides or supplements from the variable
        subsetter configuration file.

    """

    def __init__(self, variable: BaseType, cf_config: CFConfig):
        """ Extract the references contained within the variable's coordinates,
            ancillary_variables or dimensions, as returned from the initial
            pydap request. These should be augmented by information from the
            CFConfig instance passed to the class.

            Additionally, store other information required for UMM-Var record
            production in an attributes dictionary. (These attributes may not
            be an exhaustive list).

        """
        self.full_name_path = variable.attributes.get('fullnamepath')
        self.cf_config = cf_config.get_cf_attributes(self.full_name_path)
        self.group_path, self.name = self._extract_group_and_name(variable)

        self.ancillary_variables = self._get_cf_references(variable,
                                                           'ancillary_variables')
        self.coordinates = self._get_cf_references(variable, 'coordinates')
        self.subset_control_variables = self._get_cf_references(
            variable, 'subset_control_variables'
        )
        self.dimensions = self._extract_dimensions(variable)

        self.attributes = {
            'acquisition_source_name': variable.attributes.get('source'),
            'data_type': variable.dtype.name,
            'definition': variable.attributes.get('description'),
            'fill_value': variable.attributes.get('_FillValue'),
            'long_name': variable.attributes.get('long_name'),
            'offset': variable.attributes.get('offset', 0),
            'scale': variable.attributes.get('scale', 1),
            'units': variable.attributes.get('units'),
            'valid_max': variable.attributes.get('valid_max'),
            'valid_min': variable.attributes.get('valid_min')
        }

    def get_references(self) -> Set[str]:
        """ Combine the references extracted from the ancillary_variables,
            coordinates and dimensions data into a single set for VarInfo to
            use directly.

        """
        return self.ancillary_variables.union(self.coordinates,
                                              self.dimensions,
                                              self.subset_control_variables)

    def _extract_coordinates(self, variable: BaseType) -> Set[str]:
        """ Check the child elements for an Attribute element with the name
            'coordinates'. From this element, retrieve the set of coordinate
            datasets.

        """
        coordinates_string = variable.attributes.get('coordinates')

        if coordinates_string is not None:
            raw_coordinates = re.split(r'\s+|,\s*', coordinates_string)
            coordinates = self._qualify_references(raw_coordinates)
        else:
            coordinates = set()

        return coordinates

    def _get_cf_references(self, variable: BaseType,
                           attribute_name: str) -> Set[str]:
        """ Obtain the string value of a metadata attribute, which should have
            already been corrected for any known artefacts (missing or
            incorrect references). Then split this string and qualify the
            individual references.

        """
        attribute_string = self._get_cf_attribute(variable, attribute_name)
        return self._extract_references(attribute_string)

    def _get_cf_attribute(self, variable: BaseType,
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
            attribute_value = variable.attributes.get(attribute_name)

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

    def _extract_dimensions(self, variable: BaseType) -> Set[str]:
        """ Find the dimensions for the variable in question. If there are
            overriding or supplemental dimensions from the CF configuration
            file, these are used instead of, or in addtion to, the raw
            dimensions from pydap. All references are converted to absolute
            paths in the granule. A set of all fully qualified references is
            returned.

        """
        overrides = self.cf_config['cf_overrides'].get('dimensions')
        supplements = self.cf_config['cf_supplements'].get('dimensions')

        if overrides is not None:
            dimensions = re.split(r'\s+|,\s*', overrides)
        else:
            dimensions = list(variable.dimensions)

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
            name = variable.name
            group_path = None

        return group_path, name
