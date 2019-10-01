import pandas as pd
patent = pd.read_csv("patent_tsv_data/patent.tsv", sep="\t", error_bad_lines=False)
patent['now'] = pd.to_datetime('now')
patent['date'] = pd.to_datetime(patent['date'], errors='coerce')
patent = patent[patent['date'].between('1800-01-01', pd.datetime.today())]
patent['days_between_grant_and_filing'] = (patent['now'] - patent['date']).dt.days.astype('int16') 
# patent['abstract_word_count'] = patent['abstract'].str.split().str.len()
# patent = patent.dropna(subset=['abstract_word_count'])
patent = pd.get_dummies(patent, columns=['kind'])
for unknown_kind in ['kind_H', 'kind_H1', 'kind_H2', 'kind_I4', 'kind_I5', 'kind_P', 'kind_P2', 'kind_P3', 'kind_S', 'kind_S1']:
    patent = patent[patent[unknown_kind] == 0]
patent[['id', 'days_between_grant_and_filing', 'kind_A', 'kind_B1', 'kind_B2', 
        'kind_E', 'kind_E1']].to_csv("feature_data/patent_features.csv")
# patent[['id', 'days_between_grant_and_filing', 'abstract_word_count', 'kind_A', 'kind_B1', 'kind_B2', 
#         'kind_E', 'kind_E1']].to_csv("feature_data/patent_features.csv")