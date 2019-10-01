import pandas as pd
import os
import json
from joblib import dump, load

all_description_features = []
for file_name in os.listdir("feature_data/description_features/"):
    with open("feature_data/description_features/{}".format(file_name)) as f:
        description_features = json.load(f)
    for key,val in description_features.items():
        all_description_features.append([key] + val)
        
all_description_features_df = pd.DataFrame(all_description_features, columns=['patent_id', 'description_word_count', 
                                                                              'fk_score', 'fig_counts'])
all_description_features_df['patent_id'] = all_description_features_df['patent_id'].astype(str)

patent_features = pd.read_csv("feature_data/patent_features.csv", index_col=0).rename(columns={'id':'patent_id'})
patent_features['patent_id'] = patent_features['patent_id'].astype(str)

patent_data_preds = all_description_features_df.merge(patent_features, on="patent_id", how="inner")

ipcr_features = pd.read_csv("feature_data/ipcr_features.csv", index_col=0)
ipcr_features = ipcr_features.drop_duplicates(subset=['patent_id'])
ipcr_features['patent_id'] = ipcr_features['patent_id'].astype(str)

patent_data_preds = patent_data_preds.merge(ipcr_features, on="patent_id", how="inner")

patent_data_preds = patent_data_preds[['patent_id', 'days_between_grant_and_filing', 'description_word_count', 'fk_score', 'fig_counts', 'ipc_class', 'kind_A', 'kind_B1', 'kind_B2', 'kind_E', 'kind_E1', 'section_A', 'section_B', 'section_C', 'section_D', 'section_E', 'section_F', 'section_G', 'section_H', 'section_M']]

patent_data_preds['ipc_class'] = pd.to_numeric(patent_data_preds['ipc_class'], errors='coerce')
patent_data_preds = patent_data_preds.dropna(subset=['ipc_class'])
patent_data_preds['ipc_class'] = patent_data_preds['ipc_class'].astype('int')


model = load("saved_models/patent_GBC_V2.joblib")
patent_data_preds['good_claim_probability'] = model.predict_proba(patent_data_preds.iloc[:,1:])[:,1]
patent_data_preds.to_csv("pred_data/patent_data_preds_V2.csv", index=False)
patent_data_preds[['patent_id', 'good_claim_probability']].to_csv("pred_data/patent_preds_V2.csv", index=False)