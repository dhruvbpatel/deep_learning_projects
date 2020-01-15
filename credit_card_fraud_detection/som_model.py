import numpy as np 
import matplotlib.pyplot as plt 
import panda as pd

from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Credit_Card_Applications.csv')
X =dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


