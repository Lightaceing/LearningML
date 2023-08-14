import pandas as pd
import quandl
import math


df = quandl.get('WIKI/GOOGL')

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

df["hl_pct"] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100

df["pct_change"] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[["Adj. Close", "hl_pct", "pct_change", "Adj. Volume"]]

forceast = "Adj. Close"

df.fillna(-99999, inplace= True)

forceast_out = int(math.ceil(0.01 * len(df)))

df["label"] = df[forceast].shift(-forceast_out)
df.dropna(inplace=True)
print(df.head())