import pandas as pd
from utils import *
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split

from ML import *
# df = pd.read_csv('test.csv')
# print(df)
# # y = df['label']
# # X = df.drop(['label'], axis=1)
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.33, random_state=42)
# acc, pre, rec, f1, tim = ML(df, XGBClassifier(random_state=5, use_label_encoder=False))
# print("XGBoost/ acc = {}, pre = {}, rec = {}, f1 = {}, time ={}".format(acc, pre, rec, f1, tim))
# dd = pd.read_csv('dataset/MI_matrix_pd.csv', index_col=0)
# print(dd)
# from sklearn.feature_selection import mutual_info_regression
# data = pd.read_csv('dataset/concat_sarcopenia.csv', index_col=0).transpose()
# print(data)
# y = data['class']
# X = data.drop(['class'], axis=1)
# print(X)
# print(X.iloc[:, 1])
# print("-"*30)
# print(X.columns)
# mi_matrix = pd.DataFrame(index=X.columns, columns=X.columns)
# print(mi_matrix)
# print(X.columns)
# print(X.columns[1])
# print(X.columns[0])
# for i in range(0, X.shape[1]-1):
#     for j in range(i+1, X.shape[1]):
#         print("-----"+ str(i) + ", " + str(j) + "----")
#         MI = mutual_info_regression(X.iloc[:, i].values.reshape(-1,1), X.iloc[:, j].values)
#
#         mi_matrix.loc[X.columns[i], X.columns[j]] = float(MI[0])
# print(mi_matrix)
# mi_matrix.to_csv('./dataset/MI_test.csv', index=True)
# print(mi_matrix)

# print(X, y)
# print(mutual_info_classif(X, y))
data = pd.read_csv('./dataset/zoo.csv', index_col=0)
print(data)
c_m = data.corr(method='pearson')
print(c_m)
x = data.iloc[:, 0]
y = data.iloc[:, 2]
corr = np.corrcoef(x,y)[1][0]
print(corr)
# c_m.to_csv('correlation_matrix_pd.csv', index=True)
# d = pd.read_csv('./dataset/correlation_matrix_pd.csv', index_col=0)
# print(d)
# print(d.iloc[1,752])

import dask.dataframe as dd
# df = dd.read_csv('./dataset/correlation_matrix_concat_sarcopenia.csv')
# print(df)
# print(df.head())
# df =  df.set_index('Unnamed: 0', drop=True)
# print(df.head())
# import json
# start = time.time()
# with open('./dataset/XC_correlation_matrix_concat_sarcopenia.json', 'r') as file:
#     X_C = json.load(file)
# print(X_C)
# end =  time.time()
# print(end-start)
# data = pd.read_csv('zoo_custom.csv', index_col=0)
# print(data)
# acc, pre, rec, f1, tim = ML(data, XGBClassifier(random_state=5))
# print("XGBoost/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2), round(tim, 2)))
#
# acc, pre, rec, f1, tim = ML(data,
#                             MLPClassifier(random_state=5))
# print("MLPClassifier/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2), round(tim, 2)))
#
# acc, pre, rec, f1, tim = ML(data, DecisionTreeClassifier(random_state=5))
# print("DecisionTreeClassifier/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2), round(tim, 2)))
#
# acc, pre, rec, f1, tim = ML(data, SVC(random_state=5))
# print("SVC/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2), round(tim, 2)))
#
# acc, pre, rec, f1, tim = ML(data, KNeighborsClassifier())
# print("KNeighborsClassifier/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2), round(tim, 2)))