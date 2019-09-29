import pandas as pd
from readability import Readability
import sys
import csv
csv.field_size_limit(sys.maxsize)

description_features = {}
with open("detail_description_data/detail-desc-text-{}.tsv".format(sys.argv[1]),'r') as tsvin:
    tsvin = csv.reader(tsvin, delimiter='\t')
    for row in tsvin:
        patent_id = row[0]
        detail_description_text = row[1]
        try:
            description_word_count = len(detail_description_text.split())
            fig_counts = detail_description_text.count("fig.") + detail_description_text.count("figs.")
            r = Readability(detail_description_text.replace('aed-512', ''))
            fk_score = r.flesch_kincaid().score
        except:
            continue
        description_features[patent_id] = (description_word_count, fk_score, fig_counts)
with open("probability_data/description_features_{}.json".format(sys.argv[1]), 'w') as f:
    json.dump(description_features, f)
	
