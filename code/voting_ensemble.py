import glob
import pandas as pd
from collections import defaultdict
import sys
import os

bb_dict = defaultdict(list)

all_csvs = glob.glob(os.path.join(sys.argv[1], '*.txt'))
for file_ in all_csvs:
    df = pd.read_csv(file_, index_col=None, header=None)
    for i in df.iterrows():
        bb_dict[i[1].values[0]].append(i[1].values[1])

def most_common(lst):
    return max(set(lst), key=lst.count)

for bb_id, predictions in bb_dict.iteritems():
    print(str(bb_id) + ',' +  most_common(predictions))