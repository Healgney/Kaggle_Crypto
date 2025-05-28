import pandas as pd
import numpy as np

df = pd.read_parquet('/Users/wangyuhao/Desktop/drw-crypto-market-prediction/train.parquet')

df_without_inf = df.replace([np.inf, -np.inf], np.nan)
df_without_nan = df_without_inf.dropna(axis=1, how='all')
zerocol = [col for col in df_without_inf.columns if (df_without_inf[col] == 0).all()]

Data_processed = df_without_nan.drop(columns=zerocol).ffill()

Data_processed.to_csv('/Users/wangyuhao/Desktop/drw-crypto-market-prediction/trainData_afterprocessing.csv', index=False)

