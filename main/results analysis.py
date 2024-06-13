import pickle

import numpy as np
import pandas as pd

with open('results.pkl', 'rb') as Dict:
    result_Dict =pickle.load(Dict)

df = pd.DataFrame.from_dict(result_Dict)
print(df.head())
df.plot()