""" Utility classes used to extend the unittest capabilities """
from typing import Dict
import re

from pydap.model import BaseType, DatasetType
import numpy as np


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


class contains(str):
    """ Extension class that allows a 'string contains' check in a unit test
        assertion, e.g.: x.assert_called_once_with(contains('string content'))

    """
    def __eq__(self, other):
        return self.lower() in other.lower()


class matches(str):
    """ Extentsion class that allows a regular expression type check in a unit
        test assertion, e.g.: x.assert_called_once_with(matches(regex))

    """
    def __eq__(self, other):
        return re.search(self.lower(), other.lower(), re.IGNORECASE)


def write_dmr(output_dir: str, content: str):
    """ A helper function to write out the content of a `.dmr`, when the
        `harmony.util.download` function is called. This will be called as
        a side-effect to the mock for that function.

    """
    dmr_name = f'{output_dir}/downloaded.dmr'

    with open(dmr_name, 'w') as file_handler:
        file_handler.write(content)

    return dmr_name


def generate_pydap_response(variables: Dict[str, Dict[str, str]],
                            global_attributes: Dict) -> DatasetType:
    """ Create a pydap DatasetType with the requested variables and attributes,
        to mimic the output of a pydap.client.open_url request.

    """
    dataset = DatasetType()
    dataset.attributes = global_attributes

    for variable_name, variable_properties in variables.items():
        variable_attributes = variable_properties.get('attributes', {})
        variable_dimensions = variable_properties.get('dimensions', ())
        dataset[variable_name] = BaseType(variable_name, np.ones((2, 2)),
                                          attributes=variable_attributes,
                                          dimensions=variable_dimensions)

    return dataset
