import pandas as pd
import argparse
import os
from skfeature.function.information_theoretical_based import MRMR, JMI, CMIM
from sklearn.preprocessing import LabelEncoder
import numpy as np
from bench_ML import *
from utils import *

def create_folder_if_not_exists(folder_path):
    # 디렉토리 경로 확인 및 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created.")
    else:
        pass
def extract_file(filename):
    if filename == 'concat_sarcopenia':
        df = pd.read_csv('./dataset/' + filename + '.csv')
        trans_df = df.transpose()
        col_name = trans_df.loc['EnsemblID']

        df = df.drop(['EnsemblID'], axis=1)
        label = df.loc[len(df) - 1]
        data = df.iloc[:-1]
        data = data.transpose()
        df2 = pd.read_csv('./dataset/' + filename + '.csv', index_col=0).transpose()
    else:
        if filename == 'breast':
            df2 = pd.read_csv('./dataset/' + filename + '.csv', index_col=0)  # wine만 index_col=0 안함
            df2 = df2.replace("?", 0)
            df2 = df2.astype({'Bare_Nuclei': 'int'})
        elif filename == 'vehicle' or filename == 'wine':
            df2 = pd.read_csv('./dataset/' + filename + '.csv')
        elif filename == 'cancer':
            df2 = pd.read_csv('./dataset/' + filename + '.csv', index_col=0)
            df2 = df2.transpose()
        else:
            df2 = pd.read_csv('./dataset/' + filename + '.csv', index_col=0)  # wine만 index_col=0 안함
            df2 = df2.replace("?", 0)
        label = df2['class']
        col_name = df2.columns
        data = df2.drop(['class'], axis=1)
    num_cls = len(label.unique())
    return df2, data, label, col_name, num_cls


parser = argparse.ArgumentParser(description='Feature selection')

parser.add_argument('filename', type=str, default='concat_sarcopenia',
                    help='The filename will be used for feature selection')
parser.add_argument('features', type=int, default=100, help='# selected features')
parser.add_argument('method', type=str, default='JMI', help='JMI and CMIM and MRMR')
args, unknown = parser.parse_known_args()

logger = bench_setup_logger(args.filename, args.features, args.method)

df2, X, y, col_name, num_cls = extract_file(args.filename)

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
X = np.array(X)
y = np.array(y)
# print(X)
# print(y)
if args.method == 'JMI':
    selected_features = JMI.jmi(X, y, n_selected_features=args.features)
    x_ = pd.DataFrame(X[:, selected_features[0:args.features]])
    y_ = pd.DataFrame(y)
    new_df = pd.concat([x_, y_], axis=1)

elif args.method == 'MRMR':
    selected_features = MRMR.mrmr(X, y, n_selected_features=args.features)
    x_ = pd.DataFrame(X[:, selected_features[0:args.features]])
    y_ = pd.DataFrame(y)
    new_df = pd.concat([x_, y_], axis=1)

elif args.method == 'CMIM':
    selected_features = CMIM.cmim(X, y, n_selected_features=args.features)
    x_ = pd.DataFrame(X[:, selected_features[0:args.features]])
    y_ = pd.DataFrame(y)
    new_df = pd.concat([x_, y_], axis=1)

file_path = "./bench_result/" + args.filename
create_folder_if_not_exists(file_path)
new_df.to_csv("./bench_result/" + args.filename + "/" + args.filename + "_" + str(args.features) + "_" + args.method + ".csv", index=False)
data = pd.read_csv("./bench_result/" + args.filename + "/" + args.filename + "_" + str(args.features) + "_" + args.method + ".csv",header=None)
print("FS done")

acc, pre, rec, f1, tim = ML(data, XGBClassifier(random_state=5))
print_and_log(logger, "XGBoost/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2),
                                                           round(tim, 2)))

acc, pre, rec, f1, tim = ML(data,
                            MLPClassifier(random_state=5))
print_and_log(logger,
              "MLPClassifier/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2),
                                                         round(tim, 2)))

acc, pre, rec, f1, tim = ML(data, DecisionTreeClassifier(random_state=5))
print_and_log(logger,
              "DecisionTreeClassifier/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2),
                                                                  round(f1, 2), round(tim, 2)))

acc, pre, rec, f1, tim = ML(data, SVC(random_state=5))
print_and_log(logger, "SVC/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2),
                                                       round(tim, 2)))

acc, pre, rec, f1, tim = ML(data, KNeighborsClassifier())
print_and_log(logger,
              "KNeighborsClassifier/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2),
                                                                round(f1, 2), round(tim, 2)))

print_and_log(logger, "-" * 20)