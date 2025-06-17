import pandas as pd
import joblib
from sklearn.decomposition import PCA

def DataProcessing(df):
    cols_to_drop = [f"X{i}" for i in range(697, 718)]
    df.drop(columns=cols_to_drop, inplace=True)
    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df

n_components = 100
pca = PCA(n_components=n_components)

DF_train = DataProcessing(pd.read_parquet('/root/drw-crypto-market-prediction/train.parquet'))
DF_test = DataProcessing(pd.read_parquet('/root/drw-crypto-market-prediction/test.parquet'))


pca.fit(DF_train.iloc[:, :-1])

X_train_pca = pca.transform(DF_train.iloc[:, :-1])
X_test_pca = pca.transform(DF_test.iloc[:, :-1])

X_train_pca_df = pd.DataFrame(X_train_pca, index=DF_train.index,
                              columns=[f'PC{i+1}' for i in range(n_components)])
X_test_pca_df = pd.DataFrame(X_test_pca, index=DF_test.index,
                             columns=[f'PC{i+1}' for i in range(n_components)])

Train = pd.concat([X_train_pca_df, DF_train.iloc[:, -1]], axis=1)

Test = pd.concat([X_test_pca_df, DF_test.iloc[:, -1]], axis=1)

Train.to_parquet('/root/drw-crypto-market-prediction/X_train_pca.parquet')
Test.to_parquet('/root/drw-crypto-market-prediction/X_test_pca.parquet')








