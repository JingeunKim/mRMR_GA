import pandas as pd

import logging
import datetime
def bench_setup_logger(dataset, features, method):
    logger = logging.getLogger()
    log_path = './bench_logs/{:%Y%m%d}_{}_{}_{}.log'.format(datetime.datetime.now(), dataset, str(features), str(method))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def setup_logger(dataset, equation, weight, abs, method, mode):
    logger = logging.getLogger()
    if mode == 'FS':
        log_path = '../logs/{:%Y%m%d}_{}_{}_{}_{}_{}.log'.format(datetime.datetime.now(), dataset, str(equation), str(weight), str(abs), str(method))
    else:
        log_path = '../logs/Original_performance_{}.log'.format(dataset)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def print_and_log(logger, msg):
    # global logger
    print(msg)
    logger.info(msg)


def extract_file(filename):
    if filename == 'concat_sarcopenia':
        df = pd.read_csv('../dataset/' + filename + '.csv')
        trans_df = df.transpose()
        col_name = trans_df.loc['EnsemblID']

        df = df.drop(['EnsemblID'], axis=1)
        label = df.loc[len(df) - 1]
        data = df.iloc[:-1]
        data = data.transpose()
        df2 = pd.read_csv('../dataset/' + filename + '.csv', index_col=0).transpose()
    else:
        if filename == 'breast':
            df2 = pd.read_csv('../dataset/' + filename + '.csv', index_col=0)  # wine만 index_col=0 안함
            df2 = df2.replace("?", 0)
            df2 = df2.astype({'Bare_Nuclei': 'int'})
        elif filename == 'vehicle' or filename == 'wine' or filename == 'CVD' or filename == 'COPDvsILD':
            df2 = pd.read_csv('../dataset/' + filename + '.csv')
        elif filename == 'cancer':
            df2 = pd.read_csv('../dataset/' + filename + '.csv', index_col=0)
            df2 = df2.transpose()
        else:
            df2 = pd.read_csv('../dataset/' + filename + '.csv', index_col=0)  # wine만 index_col=0 안함
            df2 = df2.replace("?", 0)
        label = df2['class']
        col_name = df2.columns
        data = df2.drop(['class'], axis=1)
    num_cls = len(label.unique())
    # df2 = df2.reset_index()
    # print(df2, data, label, col_name)
    return df2, data, label, col_name, num_cls  # data가 클래스 없