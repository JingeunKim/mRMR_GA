import pandas as pd

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC

def metrics(y_test, pred):
    # print(y_test)
    # print(pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')
    f1 = f1_score(y_test, pred, average='macro')
    # print('정확도 : {0:.2f}, 정밀도 : {1:.2f}, 재현율 : {2:.2f}, f1 : {2:.2f}'.format(accuracy * 100, precision * 100, f1 * 100))
    return accuracy, precision, recall, f1

def ML(df, MLmodel):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(int)
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y = pd.DataFrame(y)
    cv = StratifiedKFold(n_splits=5, random_state=2, shuffle=True)

    time_av = []
    test_acc = []
    precision_av = []
    f1_av = []
    recall_av = []
    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        # %%time
        # print("{}st fold".format(i))
        # print(train_index, test_index)
        start_time = time.perf_counter()

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = X_train.apply(pd.to_numeric)
        y_train = y_train.apply(pd.to_numeric)
        X_test = X_test.apply(pd.to_numeric)
        y_test = y_test.apply(pd.to_numeric)
        model = MLmodel
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        accuracy, precision, recall, f1 = metrics(y_test, pred)
        # print(y_test, pred)

        test_acc.append(accuracy)
        precision_av.append(precision)
        f1_av.append(f1)
        recall_av.append(recall)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # print("CPU Time = ", elapsed_time)
        time_av.append(elapsed_time)

    return np.mean(test_acc) * 100, np.mean(precision_av) * 100, np.mean(recall_av) * 100, np.mean(f1_av)*100, np.mean(time_av)