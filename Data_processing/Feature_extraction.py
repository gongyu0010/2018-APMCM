# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing

data = pd.read_csv("data.csv")

feature_names = ["%s" % i for i in data.columns]
feature_names = feature_names[1:]

data = np.array(data)

X = data[:,1:33]
min_max_scaler = preprocessing.MinMaxScaler()
sel = VarianceThreshold(threshold=(0.03))
X = sel.fit_transform(X)
X = min_max_scaler.fit_transform(X)

Y = data[:,33:]

rf = RandomForestRegressor()
rf.fit(X, Y)

print("Features sorted by their score:")
print(sorted(zip(map(lambda x : round(x, 5), rf.feature_importances_), feature_names), reverse=True))










