import json

# Define the maps
multi_class_label_map = { 
    'a_112': 0,'e_114': 1,'i_116': 2,'la_122': 3,'le_124': 4,'li_126': 5,'ma_132': 6,'me_134': 7,'mi_136': 8, 
    'ra_142': 9,'re_144': 10,'ri_146': 11,'sa_152': 12,'se_154': 13,'si_156': 14,'ta_162': 15,'te_164': 16,'ti_166': 17,
    'a_12': 20,'e_14': 21,'i_16': 22,'la_22': 23,'le_24': 24,'li_26': 25,'ma_32': 26,'me_34': 27,'mi_36': 28, 'ra_42': 29,
    're_44': 30,'ri_46': 31,'sa_52': 32,'se_54': 33,'si_56': 34,'ta_62': 35,'te_64': 36,'ti_66': 37
}

multi_class_covert = {
    'a_112': 0,'e_114': 1,'i_116': 2,'la_122': 3,'le_124': 4,'li_126': 5,'ma_132': 6,'me_134': 7,'mi_136': 8, 
    'ra_142': 9,'re_144': 10,'ri_146': 11,'sa_152': 12,'se_154': 13,'si_156': 14,'ta_162': 15,'te_164': 16,'ti_166': 17
}

multi_class_overt = {
    'a_12': 0,'e_14': 1,'i_16': 2,'la_22': 3,'le_24': 4,'li_26': 5,'ma_32': 6,'me_34': 7,'mi_36': 8, 'ra_42': 9,
    're_44': 10,'ri_46': 11,'sa_52': 12,'se_54': 13,'si_56': 14,'ta_62': 15,'te_64': 16,'ti_66': 17
}

binary_class_label_map = {
    'a_112': 0,'e_114': 0,'i_116': 0,'la_122': 0,'le_124': 0,'li_126': 0,'ma_132': 0,'me_134': 0,'mi_136': 0, 
    'ra_142': 0,'re_144': 0,'ri_146': 0,'sa_152': 0,'se_154': 0,'si_156': 0,'ta_162': 0,'te_164': 0,'ti_166': 0,
    'a_12': 1,'e_14': 1,'i_16': 1,'la_22': 1,'le_24': 1,'li_26': 1,'ma_32': 1,'me_34': 1,'mi_36': 1, 'ra_42': 1,
    're_44': 1,'ri_46': 1,'sa_52': 1,'se_54': 1,'si_56': 1,'ta_62': 1,'te_64': 1,'ti_66': 1
}

# Combine all maps into one dictionary
all_maps = {
    "multi_class_label_map": multi_class_label_map,
    "multi_class_covert": multi_class_covert,
    "multi_class_overt": multi_class_overt,
    "binary_class_label_map": binary_class_label_map
}

# Write the combined dictionary to a single JSON file
with open('label_maps.json', 'w') as f:
    json.dump(all_maps, f)