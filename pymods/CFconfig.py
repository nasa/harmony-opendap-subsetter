import yaml
import re
"""
Single variable return:
    {
        "ancillary_variables": [],
        "coordinates": ["../latitude", "../longitude"],
        "dimensions": ["../delta_time"],
        "grid_mapping": [],
        "subset_control_variables": [{"subset_control_type": "", "segment_index_beg": "", "segment_index_cnt": ""}]
    }
Unspecified variable:
{
    "/orbit/.*": {
        "ancillary_variables": [],
	    "coordinates": ["../latitude", "../longitude"],
        "dimensions": ["../delta_time"],
	    "grid_mapping": [],
	    "subset_control_variables": []
    },
	"/quality_assessment/gt[123][l|r]/.*": {
	    "ancillary_variables": [],
		"coordinates": [],
		"dimensions": ["../delta_time", "/ds_surf_type"],
		"grid_mapping": [],
		"subset_control_variables": [{}]
	}
}
"""

class CF_Config:
    """ All arguments come from Varinfo
        Args:
        configFile - Path of YAML CF configuration file
        collection_short_name - Short name of collection from global attributes of dataset
        mission - (optional)
    """
    config = None

    def __init__(self, config_file :str, collection_short_name :str, mission :str):
        self.config_file = config_file
        self.shortname = collection_short_name
        self.config = self.read_config_file(self.config_file)
        if mission is None:
            self.mission = self.get_mission(collection_short_name)
        else:
            self.mission = mission
        global config


    def read_config_file(self, config_file):
        with open(config_file, 'r') as stream:
            global config
            config = yaml.load(stream, Loader= yaml.FullLoader)

    def get_data_organization(self, collection_short_name :str, mission :str):
        if re.match(config['CF_Supplements'][0]['Applicability']['ShortNamePath'],collection_short_name) is not None:
            return config['CF_Supplements'][0]['Data_Organization']
        elif re.match(config['CF_Supplements'][1]['Applicability']['Mission'], mission) is not None:
            return config['CF_Supplements'][1]['Data_Organization']

    def get_mission(self, short_name :str):
        for key, value in config['Mission_Shortnames'].items():
            match = re.search(value, short_name)
            if match:
                mission = key
        return mission

    def get_config_refs(self, mission: str, collection_short_name :str, variable: str):
        refs = dict()
        # reference dictionary values:
        anci_vars_vals = []
        coords_vals = []
        dim_vals = []
        grid_map_vals = []
        sub_ctrl_type_vals = []
        seg_beg_vals = []
        seg_cnt_vals = []

        if mission == config['CF_Supplements'][0]['Applicability']['Mission']:
            if variable is None:
                ref_key = config['CF_Supplements'][0]['Designated_References'][0]['Applicability']['Variable_Pattern']
            else:
                ptrn = config['CF_Supplements'][0]['Designated_References'][0]['Applicability']['Variable_Pattern']
                if (re.match(ptrn, variable) is not None):
                    dim_vals = config['CF_Overrides'][0]['Dimensions']
                if (re.match(config['CF_Supplements'][0]['Applicability']['ShortNamePath'], collection_short_name) is not None):
                    pATL = config['CF_Supplements'][0]['Designated_References']
                    if re.match(pATL[0]['Applicability']['Variable_Pattern'], variable) is not None:
                        anci_vars_vals = pATL[0]['ancillary_variables']
                    if (collection_short_name == pATL[1]['Applicability']['ShortNamePath'] and re.match(pATL[1]['Applicability']['Variable_Pattern'], variable)) is not None:
                        sub_ctrl_type_vals = pATL[1]['subset_control_type']
                    if (collection_short_name == pATL[2]['Applicability']['ShortNamePath'] and pATL[2]['subset_control_variable_type'] == 'segment_index_beg'):
                        seg_beg_vals = pATL[1]['subset_control_variables'][0]
                    if (collection_short_name == pATL[3]['Applicability']['ShortNamePath'] and pATL[3]['subset_control_variable_type'] == 'segment_index_cnt'):
                        seg_cnt_vals = pATL[1]['subset_control_variables'][1]

                    if (collection_short_name == pATL[4]['Applicability']['ShortNamePath'] and re.match(pATL[4]['Applicability']['Variable_Pattern'], variable) is not None):
                        sub_ctrl_type_vals = pATL[4]['subset_control_type']
                    if (collection_short_name == pATL[5]['Applicability']['ShortNamePath'] and pATL[5]['subset_control_variable_type'] == 'segment_index_beg'):
                        seg_beg_vals = pATL[4]['subset_control_variables'][0]
                    if (collection_short_name == pATL[6]['Applicability']['ShortNamePath'] and pATL[6]['subset_control_variable_type'] == 'segment_index_cnt'):
                        seg_cnt_vals = pATL[4]['subset_control_variables'][1]

        if mission == config['CF_Supplements'][1]['Applicability']['Mission']:
            if variable is None:
                ref_key = config['CF_Supplements'][1]['Designated_References'][0]['Applicability'][
                    'Variable_Pattern']
            else:
                pGEDI = config['CF_Supplements'][1]['Designated_References']
                # subset_control_type == coordinates
                name_ptrn = pGEDI[0]['Applicability']['ShortNamePath']
                var_ptrn = pGEDI[0]['Applicability']['Variable_Pattern']
                if (re.match(name_ptrn,collection_short_name) and re.match(var_ptrn, variable)) is not None:
                    sub_ctrl_type_vals = pGEDI[1]['subset_control_type']
                    if pGEDI[1]['subset_control_type'] == 'coordinates':
                        coords_vals = pGEDI[0]['subset_control_variables']
                elif (collection_short_name == pGEDI[1]['Applicability']['ShortNamePath']):
                    if (pGEDI[1]['subset_control_type'] == 'coordinates' and re.match(pGEDI[1]['Applicability']['Variable_Pattern'], variable)) is not None:
                        coords_vals = pGEDI[1]['subset_control_variables']
                    if (pGEDI[1]['subset_control_type'] == 'coordinates' and re.match(pGEDI[2]['Applicability']['Variable_Pattern'], variable)):
                        coords_vals = pGEDI[2]['subset_control_variables']

        refs['ancillary_variables'] = anci_vars_vals
        refs['coordinates'] = coords_vals
        refs['dimensions'] = dim_vals
        refs['grid_mapping'] = grid_map_vals
        sub_ctrl_vars = {'subset_control_type': sub_ctrl_type_vals, 'segment_index_beg': seg_beg_vals, 'segment_index_cnt': seg_cnt_vals}
        refs['subset_control_variables'] = sub_ctrl_vars

        if variable is None:
            retrefs = dict()
            retrefs[ref_key] = refs
            return retrefs

        return refs










        
        














