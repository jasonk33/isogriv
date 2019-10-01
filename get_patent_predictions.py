import pandas as pd
import os
import json
from joblib import dump, load

all_description_features = []
for file_name in os.listdir("probability_data/"):
    with open("probability_data/{}".format(file_name)) as f:
        description_features = json.load(f)
    for key,val in description_features.items():
        all_description_features.append([key] + val)
        
all_description_features_df = pd.DataFrame(all_description_features, columns=['patent_id', 'description_word_count', 
                                                                              'fk_score', 'fig_counts'])
all_description_features_df['patent_id'] = all_description_features_df['patent_id'].astype(str)

days_between_features = pd.read_csv("patent_days_between.csv", index_col=0).rename(columns={'id':'patent_id'})
days_between_features['patent_id'] = days_between_features['patent_id'].astype(str)

patent_data_preds = all_description_features_df.merge(days_between_features, on="patent_id", how="inner")
model = load("patent_GBC.joblib")
patent_data_preds['good_claim_probability'] = model.predict_proba(patent_data_preds.iloc[:,1:])[:,1]
patent_data_preds.to_csv("patent_data_preds.csv")