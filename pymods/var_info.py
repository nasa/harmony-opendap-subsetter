""" This module contains a class designed to read information from a dmr
    file. This should group the input into science variables, metadata,
    coordinates, dimesions and ancillary data sets.

"""
from typing import Dict, Set, Tuple
import re

from pydap.cas.urs import setup_session
from pydap.client import open_url
from pydap.model import BaseType
from webob.client import send_request_app
from webobtoolkit import filters

# See: https://git.earthdata.nasa.gov/projects/EMFD/repos/
#   unified-metadata-model/browse/variable/v1.5/umm-var-json-schema.json
data_types = {'byte', 'float', 'float32', 'float64', 'double', 'ubyte',
              'ushort', 'uint', 'uchar', 'string', 'char8', 'uchar8', 'short',
              'long', 'int', 'int8', 'int16', 'int32', 'int64', 'uint8',
              'uint16', 'uint32', 'uint64', 'other'}


MetaDataType = Dict

class VarInfo:
    """ A class to represent the full dataset of a granule, having read
        information from the dmr or dmrpp for it.

    """

    def __init__(self, dmr_file_url: str):
        """ Distinguish between variables containing references to other
            datasets, and those that do not. The former are considered science
            variables, providing they are not considered coordinates or
            dimensions for another variable.

            Unlike NCInfo, in SwotRepr, each variable contains references to
            their specific coordinates and dimensions, allowing the retrieval
            of all required variables for a specified list of science
            variables.

        """
        self.metadata_variables: Dict[str, Variable] = {}
        self.variables_with_coordinates: Dict[str, Variable] = {}
        self.ancillary_data: Set[str] = set()
        self.coordinates: Set[str] = set()
        self.dimensions: Set[str] = set()
        self.metadata = {}

        self._read_dataset_from_dmr(dmr_file_url)

    def _read_dataset_from_dmr(self, file_url: str):
        """ This method parses the specified dmr file. """

        """ This comment contains experimental code to get around 302 redirects
            and to implement URS authentication. The requests_application isn't
            currently functional, and requires some work.

            For an alternative methodology, see: https://github.com/pydap/pydap/issues/188

        requests_application = filters.auto_redirect_filter(
            filters.cookie_filter(
                filters.decode_filter(
                    filters.charset_filter(
                        send_request_app
                    )
                )
            )
        )

        session = setup_session('urs_username', 'urs_password',
                                check_url=file_url)
        self.dmr_input = open_url(file_url, session=session,
                                  application=requests_application)
        """
        self.dmr_input = open_url(file_url)

        for variable in self.dmr_input.values():
            # TODO: When receiving a non-flattened file, augment this to make
            # sure it is a BaseType object, if StructureType or SequenceType
            # make sure these are correctly handled.
            variable_object = Variable(variable)
            if variable_object.coordinates is not None:
                self.coordinates.update(variable_object.coordinates)
                self.variables_with_coordinates[variable_object.full_name_path] = variable_object
            else:
                self.metadata_variables[variable_object.full_name_path] = variable_object

            if variable_object.dimensions is not None:
                self.dimensions.update(variable_object.dimensions)

    def get_science_variables(self) -> Set[str]:
        """ Retrieve set of names for all variables that have coordinate
            references, that are not themselves used as dimensions, coordinates
            or ancillary date for another variable.

        """
        return (set(self.variables_with_coordinates.keys()) - self.dimensions -
                self.coordinates - self.ancillary_data)

    def get_metadata_variables(self) -> Set[str]:
        """ Retrieve set of names for all variables that do no have
            coordaintes references, that are not themselves used as dimensions,
            coordinates or ancillary data for another variable.

        """
        return (set(self.metadata_variables.keys()) - self.dimensions -
                self.coordinates - self.ancillary_data)

    def get_required_variables(self, requested_variables: Set[str]) -> Set[str]:
        """ Retrieve requested variables and recursively search for all
            associated dimension and coordinate variables. The returned set
            should be the union of the science variables, coordinates and
            dimensions.

        """
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
    """ A class to represent a single variable within the dmr or dmrpp file
        representing a granule.

    """

    def __init__(self, variable: BaseType):
        """ Create Variable object containing information compatible with
            UMM-Var records.

        """
        self.data_type = variable.dtype.name
        self.long_name = variable.attributes.get('long_name')
        self.definition = variable.attributes.get('description')
        self.scale = variable.attributes.get('scale', 1)
        self.offset = variable.attributes.get('offset', 0)
        self.acquisition_source_name = variable.attributes.get('source')
        self.units = variable.attributes.get('units')
        self.full_name_path = variable.attributes.get('fullnamepath')

        (self.group_path, self.name) = self._extract_group_and_name(variable)
        self.coordinates = self._extract_coordinates(variable)
        self.dimensions = self._extract_dimensions(variable)

        self.fill_value = variable.attributes.get('_FillValue')
        self.valid_max = variable.attributes.get('valid_max')
        self.valid_min = variable.attributes.get('valid_min')

    def _extract_coordinates(self, variable: BaseType) -> Set[str]:
        """ Check the child elements for an Attribute element with the name
            'coordinates'. From this element, retrieve the set of coordinate
            datasets.

        """
        coordinates_string = variable.attributes.get('coordinates')

        if coordinates_string is not None:
            raw_coordinates = re.split('\s+|,\s*', coordinates_string)
            coordinates = self._qualify_references(raw_coordinates)
        else:
            coordinates = set()

        return coordinates

    def _extract_dimensions(self, variable: BaseType) -> Set[str]:
        """ Find the dimensions for the variable in question. Note, this will
            only return a set of fully qualified paths to the dimension, not
            a set of UMM-Var compatible objects.

        """
        return self._qualify_references(variable.dimensions)

    def _qualify_references(self, raw_references: Tuple[str]) -> Set[str]:
        """ Take a tuple of local references to other dataset, and prepend
            the group path, if it isn't already present in the reference.

        """
        if self.group_path is not None:
            references = {self._construct_absolute_path(reference)
                          if reference.startswith('../')
                          else f'{self.group_path}/{reference}'
                          if not reference.startswith(self.group_path)
                          else reference
                          for reference in raw_references}
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
