
aa_order = list("ARNDCEQGHILKMFPSTWYVX")
AMINO_ACIDS = 'ARNDCEQGHILKMFPSTWYVX'
AMINO_ACID_IDX = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))

one_hot_encoding_scalar = {'ALA': 0.,
    'ARG': 1.,
    'ASN': 2.,
    'ASP': 3.,
    'CYS': 4.,
    'GLN': 5.,
    'GLU': 6.,
    'GLY': 7.,
    'HIS': 8.,
    'ILE': 9.,
    'LEU': 10.,
    'LYS': 11.,
    'MET': 12.,
    'PHE': 13.,
    'PRO': 14.,
    'SER': 15.,
    'THR': 16.,
    'TRP': 17.,
    'TYR': 18.,
    'VAL': 19.,
    'UNK': 20.}

three_to_one = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLU': 'E',
    'GLN': 'Q',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
    'UNK': 'X' }

standard_amino_acids = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS',
    'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL',
    'TRP', 'TYR', 'UNK' ]


one_hot_encoding = {
    'ALA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ARG': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ASN': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ASP': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'CYS': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLN': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLU': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLY': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'HIS': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ILE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LEU': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LYS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'MET': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'PHE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'PRO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'SER': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'THR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'TRP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'TYR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'VAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'UNK': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

one_hot_encoding_scalar = {'ALA': 0.,
    'ARG': 1.,
    'ASN': 2.,
    'ASP': 3.,
    'CYS': 4.,
    'GLN': 5.,
    'GLU': 6.,
    'GLY': 7.,
    'HIS': 8.,
    'ILE': 9.,
    'LEU': 10.,
    'LYS': 11.,
    'MET': 12.,
    'PHE': 13.,
    'PRO': 14.,
    'SER': 15.,
    'THR': 16.,
    'TRP': 17.,
    'TYR': 18.,
    'VAL': 19.,
    'UNK': 20.}

    

pdb_atom_types = [
    # Carbon atoms
    'C', 'CA', 'CB', 'CG', 'CD', 'CE', 'CZ', 'CH',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
    'CA1', 'CA2', 'CB1', 'CB2', 'CG1', 'CG2', 'CD1', 'CD2',
    'CE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2',
    
    # Nitrogen atoms
    'N', 'NZ', 'NH1', 'NH2', 'ND1', 'ND2', 'NE', 'NE1', 'NE2',
    
    # Oxygen atoms
    'O', 'OG', 'OG1', 'OD1', 'OD2', 'OE1', 'OE2', 'OH',
    'OXT', 'OT1', 'OT2',
    
    # Sulfur atoms
    'S', 'SG', 'SD'
    ]


column_names = ["sasa_N", "cx_N", "rd_value_N", "sasa_CA", "cx_CA", "rd_value_CA", "sasa_C", "cx_C", "rd_value_C", "sasa_O", "cx_O", "rd_value_O", 
                "cx_min", "cx_max", "cx_avg", "cx_std", "u12_dist_0", "u13_dist_0", "u14_dist_0", "u15_dist_0", "u16_dist_0", "u17_dist_0",
                "u18_dist_0", "u19_dist_0", "nearest_neighbours_cb_0", "nearest_distances_cb_0", "u20_dist_0", "u21_dist_0", "u22_dist_0",
                "u23_dist_0", "u25_dist_0", "u26_dist_0", "u27_dist_0", "u28_dist_0", "nearest_neighbours_ca_0", "nearest_distances_ca_0", 
                "t12_cos_0", "t23_cos_0", "t45_cos_0", "t56_cos_0", "t78_cos_0", "t89_cos_0", "t108_cos_0", "t811_cos_0", "t110_cos_0", 
                "t311_cos_0", "t28_cos_0", "t47_cos_0", "t69_cos_0", "t75_cos_0", "t95_cos_0", "t42_cos_0", "t62_cos_0", "t25_cos_0", "t85_cos_0", 
                "dihedral_cos_selfai_selfbi_selfbj_selfaj_0", "dihedral_sin_selfai_selfbi_selfbj_selfaj_0", "dihedral_cos_minusi_selfi_selfj_minusj_0",
                "dihedral_sin_minusi_selfi_selfj_minusj_0", "dihedral_cos_minusi_selfi_selfj_plusj_0", "dihedral_sin_minusi_selfi_selfj_plusj_0", 
                "dihedral_cos_plusi_selfi_selfj_minusj_0", "dihedral_sin_plusi_selfi_selfj_minusj_0", "dihedral_cos_plusi_selfi_selfj_plusj_0", 
                "dihedral_sin_plusi_selfi_selfj_plusj_0", "linear_foldseek_b_0", "log_foldseek_b_0", "linear_foldseek_a_0", "log_foldseek_a_0", 
                "u12_dist_1", "u13_dist_1", "u14_dist_1", "u15_dist_1", "u16_dist_1", "u17_dist_1", "u18_dist_1", "u19_dist_1", "nearest_neighbours_cb_1", 
                "nearest_distances_cb_1", "u20_dist_1", "u21_dist_1", "u22_dist_1", "u23_dist_1", "u25_dist_1", "u26_dist_1", "u27_dist_1", "u28_dist_1", 
                "nearest_neighbours_ca_1", "nearest_distances_ca_1", "t12_cos_1", "t23_cos_1", "t45_cos_1", "t56_cos_1", "t78_cos_1", "t89_cos_1", "t108_cos_1",
                "t811_cos_1", "t110_cos_1", "t311_cos_1", "t28_cos_1", "t47_cos_1", "t69_cos_1", "t75_cos_1", "t95_cos_1", "t42_cos_1", "t62_cos_1", "t25_cos_1",
                "t85_cos_1", "dihedral_cos_selfai_selfbi_selfbj_selfaj_1", "dihedral_sin_selfai_selfbi_selfbj_selfaj_1", 
                "dihedral_cos_minusi_selfi_selfj_minusj_1", "dihedral_sin_minusi_selfi_selfj_minusj_1", "dihedral_cos_minusi_selfi_selfj_plusj_1",
                "dihedral_sin_minusi_selfi_selfj_plusj_1", "dihedral_cos_plusi_selfi_selfj_minusj_1", "dihedral_sin_plusi_selfi_selfj_minusj_1",
                "dihedral_cos_plusi_selfi_selfj_plusj_1", "dihedral_sin_plusi_selfi_selfj_plusj_1", "linear_foldseek_b_1", "log_foldseek_b_1",
                "linear_foldseek_a_1", "log_foldseek_a_1", "u12_dist_2", "u13_dist_2", "u14_dist_2", "u15_dist_2", "u16_dist_2", "u17_dist_2", 
                "u18_dist_2", "u19_dist_2", "nearest_neighbours_cb_2", "nearest_distances_cb_2", "u20_dist_2", "u21_dist_2", "u22_dist_2", "u23_dist_2",
                "u25_dist_2", "u26_dist_2", "u27_dist_2", "u28_dist_2", "nearest_neighbours_ca_2", "nearest_distances_ca_2", "t12_cos_2", "t23_cos_2", 
                "t45_cos_2", "t56_cos_2", "t78_cos_2", "t89_cos_2", "t108_cos_2", "t811_cos_2", "t110_cos_2", "t311_cos_2", "t28_cos_2", "t47_cos_2", 
                "t69_cos_2", "t75_cos_2", "t95_cos_2", "t42_cos_2", "t62_cos_2", "t25_cos_2", "t85_cos_2", "dihedral_cos_selfai_selfbi_selfbj_selfaj_2", 
                "dihedral_sin_selfai_selfbi_selfbj_selfaj_2", "dihedral_cos_minusi_selfi_selfj_minusj_2", "dihedral_sin_minusi_selfi_selfj_minusj_2", 
                "dihedral_cos_minusi_selfi_selfj_plusj_2", "dihedral_sin_minusi_selfi_selfj_plusj_2", "dihedral_cos_plusi_selfi_selfj_minusj_2", "dihedral_sin_plusi_selfi_selfj_minusj_2",
                "dihedral_cos_plusi_selfi_selfj_plusj_2", "dihedral_sin_plusi_selfi_selfj_plusj_2", "linear_foldseek_b_2", "log_foldseek_b_2", "linear_foldseek_a_2", "log_foldseek_a_2",
                "u12_dist_3", "u13_dist_3", "u14_dist_3", "u15_dist_3", "u16_dist_3", "u17_dist_3", "u18_dist_3", "u19_dist_3", "nearest_neighbours_cb_3", "nearest_distances_cb_3", 
                "u20_dist_3", "u21_dist_3", "u22_dist_3", "u23_dist_3", "u25_dist_3", "u26_dist_3", "u27_dist_3", "u28_dist_3", "nearest_neighbours_ca_3", "nearest_distances_ca_3", 
                "t12_cos_3", "t23_cos_3", "t45_cos_3", "t56_cos_3", "t78_cos_3", "t89_cos_3", "t108_cos_3", "t811_cos_3", "t110_cos_3", "t311_cos_3", "t28_cos_3", "t47_cos_3", "t69_cos_3",
                "t75_cos_3", "t95_cos_3", "t42_cos_3", "t62_cos_3", "t25_cos_3", "t85_cos_3", "dihedral_cos_selfai_selfbi_selfbj_selfaj_3", "dihedral_sin_selfai_selfbi_selfbj_selfaj_3", 
                "dihedral_cos_minusi_selfi_selfj_minusj_3", "dihedral_sin_minusi_selfi_selfj_minusj_3", "dihedral_cos_minusi_selfi_selfj_plusj_3", "dihedral_sin_minusi_selfi_selfj_plusj_3", 
                "dihedral_cos_plusi_selfi_selfj_minusj_3", "dihedral_sin_plusi_selfi_selfj_minusj_3", "dihedral_cos_plusi_selfi_selfj_plusj_3", "dihedral_sin_plusi_selfi_selfj_plusj_3", 
                "linear_foldseek_b_3", "log_foldseek_b_3", "linear_foldseek_a_3", "log_foldseek_a_3", "u12_dist_4", "u13_dist_4", "u14_dist_4", "u15_dist_4", "u16_dist_4", "u17_dist_4", 
                "u18_dist_4", "u19_dist_4", "nearest_neighbours_cb_4", "nearest_distances_cb_4", "u20_dist_4", "u21_dist_4", "u22_dist_4", "u23_dist_4", "u25_dist_4", "u26_dist_4",
                "u27_dist_4", "u28_dist_4", "nearest_neighbours_ca_4", "nearest_distances_ca_4", "t12_cos_4", "t23_cos_4", "t45_cos_4", "t56_cos_4", "t78_cos_4", "t89_cos_4", "t108_cos_4",
                "t811_cos_4", "t110_cos_4", "t311_cos_4", "t28_cos_4", "t47_cos_4", "t69_cos_4", "t75_cos_4", "t95_cos_4", "t42_cos_4", "t62_cos_4", "t25_cos_4", "t85_cos_4", 
                "dihedral_cos_selfai_selfbi_selfbj_selfaj_4", "dihedral_sin_selfai_selfbi_selfbj_selfaj_4", "dihedral_cos_minusi_selfi_selfj_minusj_4", "dihedral_sin_minusi_selfi_selfj_minusj_4",
                "dihedral_cos_minusi_selfi_selfj_plusj_4", "dihedral_sin_minusi_selfi_selfj_plusj_4", "dihedral_cos_plusi_selfi_selfj_minusj_4", "dihedral_sin_plusi_selfi_selfj_minusj_4", 
                "dihedral_cos_plusi_selfi_selfj_plusj_4", "dihedral_sin_plusi_selfi_selfj_plusj_4", "linear_foldseek_b_4", "log_foldseek_b_4", "linear_foldseek_a_4", "log_foldseek_a_4",
                "cos_ca_k0_and_k1", "cos_cb_k0_and_k1", "cos_ca_k0_and_k2", "cos_cb_k0_and_k2", "cos_ca_k0_and_k3", "cos_cb_k0_and_k3", "cos_ca_k0_and_k4", "cos_cb_k0_and_k4", 
                "cos_ca_k1_and_k2", "cos_cb_k1_and_k2", "cos_ca_k1_and_k3", "cos_cb_k1_and_k3", "cos_ca_k1_and_k4", "cos_cb_k1_and_k4", "cos_ca_k2_and_k3", "cos_cb_k2_and_k3", 
                "cos_ca_k2_and_k4", "cos_cb_k2_and_k4", "cos_ca_k3_and_k4", "cos_cb_k3_and_k4"]




remove_indices_291 = [24, 34,77, 87, 130, 140, 
                  183, 193, 236, 246]
remove_set_291 = remove_indices_291
keep_indices_291 = [i for i in range(301) if i not in remove_set_291]
column_names_291 = [column_names[i] for i in keep_indices_291]
angle_columns_291 = {}
for i, cn in enumerate(column_names_291):
    if '_cos' in cn or '_sin' in cn: 
        angle_columns_291[cn] = i
cos_sine_columns_291 = {}
for i, cn in enumerate(column_names_291):
    if 'dihedral_cos_' in cn:
        assert column_names_291[i].replace('_cos', '') == column_names_291[i+1].replace('_sin', ''), (column_names_291[i], column_names_291[i+1])
        cos_sine_columns_291[cn+'/'+column_names_291[i+1]] = [i, i+1]
not_angle_columns_291 = {}
for i, cn in enumerate(column_names_291):
    if '_cos' in cn or '_sin' in cn:
        continue 
    else:    
        not_angle_columns_291[cn] = i



remove_indices_281 = [24, 25, 34, 35, 77, 78, 87, 88, 130, 131, 140, 141, 
                  183, 184, 193, 194, 236, 237, 246, 247]
remove_set_281 = remove_indices_281
keep_indices_281 = [i for i in range(301) if i not in remove_set_281]
column_names_281 = [column_names[i] for i in keep_indices_281]
angle_columns_281 = {}
for i, cn in enumerate(column_names_281):
    if '_cos' in cn or '_sin' in cn: 
        angle_columns_281[cn] = i
cos_sine_columns_281 = {}
for i, cn in enumerate(column_names_281):
    if 'dihedral_cos_' in cn:
        assert column_names_281[i].replace('_cos', '') == column_names_281[i+1].replace('_sin', ''), (column_names_281[i], column_names_281[i+1])
        cos_sine_columns_281[cn+'/'+column_names_281[i+1]] = [i, i+1]
not_angle_columns_281 = {}
for i, cn in enumerate(column_names_281):
    if '_cos' in cn or '_sin' in cn:
        continue 
    else:    
        not_angle_columns_281[cn] = i


test_unique_taxonomies = [44252, 1660138, 1817866, 1137991, 2626420, 55423, 1421003, 2293317, 485916, 133453, 47735, 221109, 1797484, 60137, 2135497, 
                        985250, 9702, 767770, 2662433, 7721, 45076, 1173111, 1437875, 1852934, 316273, 1156392, 538378, 62609, 538948, 6731, 1667357,
                        1712028, 8690, 1131320, 2072420, 2735434, 1200346, 1121429, 2632609, 1259464, 1811678, 2049881, 229533, 103733, 2587048, 
                        60542, 306902, 580340, 742727, 1262877, 997887, 568106, 1896213, 558541, 544712, 2292351, 266128, 1904754, 1797220, 1881110, 
                        592308, 537010, 121277, 708627, 360107, 1577904, 2724121, 374981, 30460, 1121400, 1982044, 2070369, 1262885, 2013096, 86665, 
                        645990, 2654249, 1742972, 1797649, 289370, 1897032, 444923, 2292240, 216597, 1698275, 1679167, 518642, 1798676, 1348852, 64969,
                        2135670, 1811715, 630973, 1262857, 1456404, 1352941, 1450533, 749927, 1799661, 1802586, 1902579, 658225, 1619010, 673271, 
                        1265817, 1801856, 93944, 444093, 361101, 2552981, 1125979, 3764, 1761016, 742738, 280332, 158500, 1703944, 1805265, 2293254,
                        253239, 1172190, 1423760, 942865, 1869307, 2584122, 1172562, 1184607, 2340917, 1797982, 1907941, 1965623, 2029980, 1955414,
                        198613, 1797673, 1904965, 153971, 858, 2778370, 35783, 1325107, 1909303, 2763107, 1353, 1802663, 1798388, 475083, 1850348,
                        2496849, 2501295, 2635087, 43595, 118163, 1126616, 442341, 1805399, 195709, 2607614, 1397528, 1583341, 1797242, 1797290, 
                        1884310, 1848292, 1857892, 2806336, 446469, 2590670, 1181504, 645991, 1310605, 1834076, 1159016, 1804986, 52768, 46158,
                        2678885, 1280952, 1542388, 1080709, 2013818, 317018, 1797844, 1895781, 2596826, 2482727, 2282131, 1797183, 1089443,
                        1276229, 415000]


# Grantham Polarity
grantham_polarity = {
    "ALA": 8.100, "ARG": 10.500, "ASN": 11.600, "ASP": 13.000, "CYS": 5.500,
    "GLN": 10.500, "GLU": 12.300, "GLY": 9.000, "HIS": 10.400, "ILE": 5.200,
    "LEU": 4.900, "LYS": 11.300, "MET": 5.700, "PHE": 5.200, "PRO": 8.000,
    "SER": 9.200, "THR": 8.600, "TRP": 5.400, "TYR": 6.200, "VAL": 5.900
}

# Miyazawa hydrophobicity scale
miyazawa_hydrophobicity = {
    "ALA": 5.330, "ARG": 4.180, "ASN": 3.710, "ASP": 3.590, "CYS": 7.930,
    "GLN": 3.870, "GLU": 3.650, "GLY": 4.480, "HIS": 5.100, "ILE": 8.830,
    "LEU": 8.470, "LYS": 2.950, "MET": 8.950, "PHE": 9.030, "PRO": 3.870,
    "SER": 4.090, "THR": 4.490, "TRP": 7.660, "TYR": 5.890, "VAL": 7.630
}

# Bulkiness
bulkiness = {
    "ALA": 11.500, "ARG": 14.280, "ASN": 12.820, "ASP": 11.680, "CYS": 13.460,
    "GLN": 14.450, "GLU": 13.570, "GLY": 3.400, "HIS": 13.690, "ILE": 21.400,
    "LEU": 21.400, "LYS": 15.710, "MET": 16.250, "PHE": 19.800, "PRO": 17.430,
    "SER": 9.470, "THR": 15.770, "TRP": 21.670, "TYR": 18.030, "VAL": 21.570
}