from logging import Logger
from unittest import TestCase
import yaml

from pymods.cf_config import CFConfig


# Mock yaml.load for whole class
class TestCFConfig(TestCase):
    """ Perform unit tests to establish the CFConfig class can be successfully
        instantiated, with the expected class attributes and methods. Also
        test public methods within the class to ensure they return the expected
        results.

    """
    @classmethod
    def setUpClass(cls):
        """ Set attributes for the class that can be shared between tests. """
        cls.test_config = 'tests/unit/data/test_config.yml'
        cls.mission = 'FakeSat'
        cls.short_name = 'FAKE99'
        cls.excluded_science_variables = {'/exclude_one/.*',
                                          '/exclude_two/.*',
                                          '/exclude_three/.*'}
        cls.required_variables = {'/required_group/.*'}
        cls.global_overrides = {'global_override': 'GLOBAL'}
        cls.global_supplements = {'fakesat_global_supplement': 'fakesat value'}
        cls.cf_overrides = {
            '.*': {'collection_override': 'collection value'},
            '/group/.*': {'group_override': 'group value'},
            '/group/variable': {'variable_override': 'variable value'}
        }
        cls.cf_supplements = {
            '.*': {'collection_supplement': 'FAKE99 supplement'},
            '/group4/.*': {'group_supplement': 'FAKE99 group4'}
        }

    def test_instantiation(self):
        """ Ensure the attributes of an object are set upon class
        instantiation. This should include mission, short_name and information
        from the configuration file. It should also exclude all attributes with
        applicabilities that do not match the supplied mission and short name.

        """
        config = CFConfig(self.mission, self.short_name, self.test_config)

        self.assertEqual(self.mission, config.mission)
        self.assertEqual(self.short_name, config.short_name)

        self.assertCountEqual(self.excluded_science_variables,
                              config.excluded_science_variables)

        self.assertCountEqual(self.required_variables,
                              config.required_variables)

        self.assertEqual(self.global_overrides, config.global_overrides)
        self.assertEqual(self.global_supplements, config.global_supplements)

        # The attributes below are protected-access within the class, however,
        # this test should still check they only contain the expected items.
        self.assertEqual(self.cf_overrides, config._cf_overrides) # pylint: disable=W0212
        self.assertEqual(self.cf_supplements, config._cf_supplements) # pylint: disable=W0212

    def test_get_cf_attributes_all(self):
        """ Ensure the CFConfig.get_cf_references method returns all the
            overriding and supplemental references from the class, in
            dictionaries that are keyed on the variable pattern.

        """
        config = CFConfig(self.mission, self.short_name, self.test_config)
        self.assertEqual(config.get_cf_attributes(),
                         {'cf_overrides': self.cf_overrides,
                          'cf_supplements': self.cf_supplements})

    def test_get_cf_attributes_variable(self):
        """ Ensure the CFConfig.get_cf_references method returns all overriding
            and supplemental attributes where the variable pattern matches the
            supplied variable name. If multiple patterns match the variable
            name, then all attributes from those patterns should be combined
            into a single output dictionary.

        """
        collection_overrides = {'collection_override': 'collection value'}
        group_overrides = {'collection_override': 'collection value',
                           'group_override': 'group value'}
        variable_overrides = {'collection_override': 'collection value',
                              'group_override': 'group value',
                              'variable_override': 'variable value'}

        collection_supplements = {'collection_supplement': 'FAKE99 supplement'}
        group4_supplements = {'collection_supplement': 'FAKE99 supplement',
                              'group_supplement': 'FAKE99 group4'}

        test_args = [
            ['Collection only', 'random_variable', collection_overrides,
             collection_supplements],
            ['Group overrides', '/group/random', group_overrides,
             collection_supplements],
            ['Variable overrides', '/group/variable', variable_overrides,
             collection_supplements],
            ['Group supplements', '/group4/variable', collection_overrides,
             group4_supplements]]

        config = CFConfig(self.mission, self.short_name, self.test_config)

        for description, variable, overrides, supplements in test_args:
            with self.subTest(description):
                self.assertEqual(config.get_cf_attributes(variable),
                                 {'cf_overrides': overrides,
                                  'cf_supplements': supplements})
