import pandas as pd

def fileloader(path, file, req_cols, dtypes):
    return pd.read_csv(f"{path}/{file}", delimiter=',', quotechar='"', low_memory=False, usecols=req_cols, dtype=dtypes)
