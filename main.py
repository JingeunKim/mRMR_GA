import argparse
import time
from utils import *
import feature_selection_GA
import pandas as pd
from ML import *

parser = argparse.ArgumentParser(description='Feature selection')

parser.add_argument('filename', type=str, default='concat_sarcopenia',
                    help='The filename will be used for feature selection')
parser.add_argument('human_number', type=int, default=100, help='Initial population size')
parser.add_argument('gene_number', type=int, default=100, help='The number of selected features')
parser.add_argument('generation_num', type=int, default=100, help='The number of generations')
parser.add_argument('weight', type=float, default=1.0, help='weight')
parser.add_argument('equation', type=str, default='mRMR', help='version of fitness function')
parser.add_argument('abs', type=str, default='XC', help='select XY or XC or all')
parser.add_argument('method', type=str, default='Corr', help='Corr or MI')
parser.add_argument('mode', type=str, default='FS', help='OG and FS')

args, unknown = parser.parse_known_args()

logger = setup_logger(args.filename, args.equation, args.weight, args.abs, args.method, args.mode)
if args.mode == 'FS':
    print_and_log(logger, "file name = " + str(args.filename))
    print_and_log(logger, "population size = " + str(args.human_number))
    print_and_log(logger, "% of selected features = " + str(args.gene_number))
    print_and_log(logger, "generation number = " + str(args.generation_num))
    print_and_log(logger, "equation = " + str(args.equation))
    print_and_log(logger, "weight = " + str(args.weight))
    print_and_log(logger, "abs = " + str(args.abs))
    print_and_log(logger, "method = " + str(args.method))
    start = time.process_time()

    df, data, label, col_name, cls_name = extract_file(args.filename)

    ga = feature_selection_GA.GA(data.shape[0], data.shape[1], args.filename, args.human_number, round(args.gene_number),
                                 args.generation_num, args.weight, args.equation, args.abs, args.method)
    aa = feature_selection_GA.GA.evolve(ga)
    t = time.process_time() - start

    print_and_log(logger, "process time = " + str(t) + 'sec')

    print_and_log(logger, "-" * 20)
    data = pd.read_csv(
        "../result/After_GA/GA_" + str(args.filename) + "_" + str(args.human_number) + "_" + str(
                    round(args.gene_number)) + "_" + str(args.generation_num) + "_" + str(args.weight) + "_" + args.equation + "_" + args.abs + "_" + args.method + ".csv",
        delimiter='\t',
        header=None, index_col=0)
    index = data.shape[1]
    data = data.drop(index, axis=1).transpose()

    acc, pre, rec, f1, tim = ML(data, XGBClassifier(random_state=5))
    print_and_log(logger, "XGBoost/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2), round(tim, 2)))

    acc, pre, rec, f1, tim = ML(data,
                                MLPClassifier(random_state=5))
    print_and_log(logger, "MLPClassifier/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2), round(tim, 2)))

    acc, pre, rec, f1, tim = ML(data, DecisionTreeClassifier(random_state=5))
    print_and_log(logger,
                  "DecisionTreeClassifier/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2), round(tim, 2)))

    acc, pre, rec, f1, tim = ML(data, SVC(random_state=5))
    print_and_log(logger, "SVC/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2), round(tim, 2)))

    acc, pre, rec, f1, tim = ML(data, KNeighborsClassifier())
    print_and_log(logger,
                  "KNeighborsClassifier/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2), round(rec, 2), round(f1, 2), round(tim, 2)))

    print_and_log(logger, "-" * 20)
else:
    print_and_log(logger, "file name = " + str(args.filename))
    data, df, label, col_name, cls_name = extract_file(args.filename)
    acc, pre, rec, f1, tim = ML(data, XGBClassifier(random_state=5))
    print_and_log(logger,
                  "XGBoost/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2),
                                                                                    round(rec, 2), round(f1, 2),
                                                                                    round(tim, 2)))

    acc, pre, rec, f1, tim = ML(data,
                                MLPClassifier(random_state=5))
    print_and_log(logger,
                  "MLPClassifier/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2),
                                                                                          round(rec, 2), round(f1, 2),
                                                                                          round(tim, 2)))

    acc, pre, rec, f1, tim = ML(data, DecisionTreeClassifier(random_state=5))
    print_and_log(logger,
                  "DecisionTreeClassifier/ {}, {}, {}, {}, {}".format(round(acc, 2),
                                                                                                   round(pre, 2),
                                                                                                   round(rec, 2),
                                                                                                   round(f1, 2),
                                                                                                   round(tim, 2)))

    acc, pre, rec, f1, tim = ML(data, SVC(random_state=5))
    print_and_log(logger, "SVC/ {}, {}, {}, {}, {}".format(round(acc, 2), round(pre, 2),
                                                                                        round(rec, 2), round(f1, 2),
                                                                                        round(tim, 2)))

    acc, pre, rec, f1, tim = ML(data, KNeighborsClassifier())
    print_and_log(logger,
                  "KNeighborsClassifier/ {}, {}, {}, {}, {}".format(round(acc, 2),
                                                                                                 round(pre, 2),
                                                                                                 round(rec, 2),
                                                                                                 round(f1, 2),
                                                                                                 round(tim, 2)))

    print_and_log(logger, "-" * 20)