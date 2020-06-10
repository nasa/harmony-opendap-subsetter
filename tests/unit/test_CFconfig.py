from logging import Logger
from unittest import TestCase
import yaml
from pymods.var_info import VarInfo
from pymods.CFconfig import CF_Config


class TestCFConfig(TestCase):

    def setUp(cls):
        cls.conf_file = "/Users/vskorole/PyCharmProjects/var_subsetter/VarSubsetter_Config.yml"
        cls.config = CF_Config(cls.conf_file, 'ATL03', None)

    def test_get_mission(self): # and mission
        short_name = 'ATL03'
        self.assertEqual(self.config.get_mission(short_name), 'ICESat2')

    def test_get_data_organization(self):
        mission = 'GEDI'
        short_name = 'GEDI_L1A'
        self.assertEqual(self.config.get_data_organization(mission,short_name),'h5_trajectory')

    def test_get_config_refs(self):
        mission = 'ICESat2'
        short_name = 'ATL03'
        var = 'quality_assessment_qa_granule_pass_fail'
        conf_refs = self.config.get_config_refs(mission, short_name, var)
        result = {'ancillary_variables': [], 'coordinates': [], 'dimensions': [], 'grid_mapping': [], 'subset_control_variables': {'subset_control_type': [], 'segment_index_beg': '../geolocation/ph_index_beg', 'segment_index_cnt': '../geolocation/segment_ph_cnt'}}
        self.assertEqual(conf_refs,result)


