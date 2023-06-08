import numpy as np
import pandas as pd
import joblib

# dataset = pd.read_csv('Position_Salaries.csv')
# X = dataset.iloc[: , 1:-1].values
# y = dataset.iloc[: , -1].values

# y = y.reshape(len(y),1)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X = sc.fit_transform(X)
# sc1 = StandardScaler()
# y = sc1.fit_transform(y)

# from sklearn.svm import SVR
# regressor = SVR(kernel='rbf')
# regressor.fit(X , y)

# print(sc1.inverse_transform(regressor.predict(sc.transform([[6.5]])).reshape(-1 , 1)))

dataset1 = pd.read_csv('Social_Network_Ads.csv')
X1 = dataset1.iloc[: , :-1].values
y1 = dataset1.iloc[: , -1].values

# joblib.dump(regressor , "model.pkl")

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1 = sc.fit_transform(X1)

joblib.dump(sc, "scaler.pkl")

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X1 , y1)

joblib.dump(regressor, "regressor.pkl")

print(regressor.predict(sc.transform([[19,19000]])))
