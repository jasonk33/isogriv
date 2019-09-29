import pandas as pd
patent = pd.read_csv("patent.tsv", sep="\t", error_bad_lines=False)
patent['now'] = pd.to_datetime('now')
patent['date'] = pd.to_datetime(patent['date'], errors='coerce')
patent = patent[patent['date'].between('1800-01-01', pd.datetime.today())]
patent['days_between_grant_and_filing'] = (patent['now'] - patent['date']).dt.days.astype('int16') 
patent[['id', 'days_between_grant_and_filing']].to_csv("patent_days_between.csv")