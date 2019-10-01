import pandas as pd
ipcr = pd.read_csv("patent_tsv_data/ipcr.tsv", sep="\t", error_bad_lines=False)
ipcr = pd.get_dummies(ipcr, columns=['section'])
for unknown_kind in ['section_0', 'section_1',
       'section_2', 'section_3', 'section_4', 'section_5', 'section_6',
       'section_8', 'section_9', 'section_?', 'section_I', 'section_J', 'section_K', 'section_L', 'section_N', 'section_O', 'section_P', 'section_Q',
       'section_R', 'section_S', 'section_T', 'section_U', 'section_V',
       'section_W', 'section_X', 'section_Y', 'section_Z', 'section_b',
       'section_c', 'section_e', 'section_g', 'section_h']:
    ipcr = ipcr[ipcr[unknown_kind] == 0]
ipcr[['patent_id', 'ipc_class', 'section_A', 'section_B', 'section_C', 'section_D', 'section_E', 'section_F', 'section_G', 
        'section_H', 'section_M']].to_csv("feature_data/ipcr_features.csv")