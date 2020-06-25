from typing import Dict
import re
import yaml


class CFConfig:
    """ This class should read the main configuration file,
        var_subsetter_config.yml, which defines overriding values and
        supplements for the attributes stored in fields such as
        ancillary_variables, or dimensions.

        Given a mission and collection short name, upon instantiation, the
        object should only retain information relevant to that specific
        collection.

    """
    def __init__(self, mission: str, collection_short_name: str,
                 config_file: str = 'pymods/var_subsetter_config.yml'):
        """ Set supplied class attributes. Then read the designated
            configuration file to obtain mission and short name specific
            attributes.

        """
        self.config_file = config_file
        self.mission = mission
        self.short_name = collection_short_name

        self._cf_overrides = {}
        self._cf_supplements = {}
        self.excluded_science_variables = set()
        self.required_variables = set()
        self.global_supplements = {}
        self.global_overrides = {}

        if self.mission is not None:
            self._read_config_file()

    def _read_config_file(self):
        """ Open the main configuration YAML file and extract only those
            parts of it pertaining to the mission and collection specified
            upon instantiating the class.

        """
        with open(self.config_file, 'r') as file_handler:
            config = yaml.load(file_handler, Loader=yaml.FullLoader)

        self.excluded_science_variables = {
            pattern
            for item in config['Excluded_Science_Variables']
            if self._is_applicable(item['Applicability'].get('Mission'),
                                   item['Applicability'].get('ShortNamePath'))
            for pattern in item['Variable_Pattern']
        }

        self.required_variables = {
            pattern
            for item in config['Required_Fields']
            if self._is_applicable(item['Applicability'].get('Mission'),
                                   item['Applicability'].get('ShortNamePath'))
            for pattern in item['Variable_Pattern']
        }

        self.global_supplements = {
            attribute['Name']: attribute['Value']
            for item in config['CF_Supplements']
            if self._is_applicable(item['Applicability'].get('Mission'),
                                   item['Applicability'].get('ShortNamePath'))
            for attribute in item.get('Global_Attributes', [])
        }

        self.global_overrides = {
            attribute['Name']: attribute['Value']
            for item in config['CF_Overrides']
            if self._is_applicable(item['Applicability'].get('Mission'),
                                   item['Applicability'].get('ShortNamePath'))
            for attribute in item.get('Global_Attributes', [])
        }

        for override in config['CF_Overrides']:
            self._process_cf_item(override, self._cf_overrides)

        for supplement in config['CF_Supplements']:
            self._process_cf_item(supplement, self._cf_supplements)

    def _is_applicable(self, mission: str, short_name: str = None) -> bool:
        """ Given a mission, and optionally also a collection short name, of an
            applicability within the configuration file, check for a match
            against the mission and short name specified when instantiating the
            class object.

        """
        mission_matches = re.match(mission, self.mission) is not None

        short_name_matches = (
            short_name is None or
            re.match(short_name, self.short_name) is not None
        )

        return mission_matches and short_name_matches

    def _process_cf_item(self,
                         cf_item: Dict,
                         results: Dict[str, Dict],
                         input_mission: str = None,
                         input_short_name: str = None):
        """ Process a single block in the CF overrides or CF supplements region
            of the configuration file. First check that the applicability
            matches the mission and short name for the class. Next, check
            for a variable pattern. This is indicative of there being
            overriding or supplemental attributes in this list item.
            Assign any information to the results dictionary, with a key of
            that variable pattern. Lastly, check for any nested references,
            which are child blocks to be processed in the same way. The mission
            and short name from this block are passed to all children, as they
            may not both be defined, due to assumed inheritance.

        """
        mission = cf_item['Applicability'].get('Mission') or input_mission
        short_name = cf_item['Applicability'].get('ShortNamePath') or input_short_name

        if mission is not None and self._is_applicable(mission, short_name):
            # Some outer Applicability items have attributes, but no
            # variable path - the assumption here is that the applicability is
            # to all variables (see ICESat2 dimensions override, SPL4.* and
            # SPL3FTA grid_mapping overrides)
            pattern = cf_item['Applicability'].get('Variable_Pattern', '.*')

            if 'Attributes' in cf_item:
                results[pattern] = self._create_attributes_object(cf_item)

            cf_references = cf_item.get('Applicability_Group', [])

            for cf_reference in cf_references:
                self._process_cf_item(cf_reference, results, mission,
                                      short_name)

    @staticmethod
    def _create_attributes_object(cf_item: Dict) -> Dict[str, str]:
        """ Construct a dictionary object containing all contained attributes,
            which are specified as list items with Name and Value keys.

        """
        return {attribute['Name']: attribute['Value']
                for attribute in cf_item.get('Attributes', {})}

    def get_cf_attributes(self, variable: str = None) -> Dict[str, Dict[str, str]]:
        """ Return the CF overrides and supplements that match a given
            variable. If a variable is not specified, then return all overrides
            and supplements. If there are no overrides or supplements, then
            empty dictionaries will be returned instead.

        """
        if variable is not None:
            cf_overrides = self._get_matching_attributes(self._cf_overrides,
                                                         variable)
            cf_supplements = self._get_matching_attributes(self._cf_supplements,
                                                           variable)
        else:
            cf_overrides = self._cf_overrides
            cf_supplements = self._cf_supplements

        return {'cf_overrides': cf_overrides, 'cf_supplements': cf_supplements}

    @staticmethod
    def _get_matching_attributes(cf_references: Dict[str, Dict[str, str]],
                                 variable: str) -> Dict[str, str]:
        """ Iterate through either the self._cf_supplements or
            self._cf_overrides and extract a dictionary that combines all
            applicable attributes that apply to the specified variable. If
            there are conflicting values for the same attribute, only the last
            value will be returned for that attribute.

        """
        references = {}
        for pattern, attributes in cf_references.items():
            if re.match(pattern, variable) is not None:
                references.update(attributes)

        return references
