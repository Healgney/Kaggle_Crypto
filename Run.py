from model import *
from main import MyModel,Mydataset
import numpy as np
import pandas as pd

Data = pd.read_csv('/Users/wangyuhao/Desktop/drw-crypto-market-prediction/trainData_afterprocessing.csv')

Mydata = Mydataset(Data.iloc[:,:-1],Data.iloc[:,-1])
batch_size = 12
window_size = 12
feature_number = len(Data.iloc[:,:-1])
RATtransformer = make_model(batch_size, window_size, feature_number)

