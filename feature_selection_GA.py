import random
from scipy import stats as sp
from scipy.optimize import fminbound
from utils import *
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from dask import dataframe as dd

random.seed(10)
np.random.seed(10)

class GA():
    def __init__(self, row, col, filename, human_number, gene_number, generation_num, weight, equation, abs, method):
        self.human_number = human_number
        self.gene_number = gene_number
        self.generation_num = generation_num
        self.filename = filename
        self.row = row
        self.col = col
        self.weight = weight
        self.equation = equation
        self.abs = abs
        self.method = method
        # print(row, col)

    def initialization(self):
        human = np.zeros((self.human_number, self.col))
        for j in range(self.human_number):
            createNum = self.createNumber()
            for i in range(self.gene_number):
                human[j][createNum[i]] = 1
        return human

    def createNumber(self):
        nb = []
        rnum = random.randint(0, self.col - 1)
        for i in range(self.gene_number):
            while rnum in nb:
                rnum = random.randint(0, self.col - 1)
            nb.append(rnum)
        return nb

    def polyserial_correlation(self, continuous, ordinal):
        """
        Computes the polyserial correlation.

        Estimates the correlation value based on a bivariate
        normal distribution.

        :
            continuous: Continuous Measurement
            ordinal: Ordinal Measurement

        Returns:
            polyserial_correlation: converged value

        Notes:
            User must handle missing data
        """
        # Get the number of ordinal values
        values, counts = np.unique(ordinal, return_counts=True)

        # Compute the thresholds (tau's) inverse of survival function
        thresholds = sp.norm.isf(1 - counts.cumsum() / counts.sum())[:-1]
        # inverse values of the normal distribution function evaluated at the cumulative marginal proportions of Y are taken as estimates of the thresholds taul, tau2 ..... taui

        # Standardize the continuous variable
        standardized_continuous = ((continuous - continuous.mean())
                                   / continuous.std(ddof=1))

        # print(continuous)

        def _min_func(correlation):
            denominator = np.sqrt(1 - correlation * correlation)
            k = standardized_continuous * correlation
            log_likelihood = 0

            for ndx, value in enumerate(values):
                mask = ordinal == value

                if ndx == 0:
                    numerator = thresholds[ndx] - k[mask]
                    probabilty = sp.norm.cdf(numerator / denominator)  # cumulative distribution function

                elif ndx == (values.size - 1):
                    numerator = thresholds[ndx - 1] - k[mask]
                    probabilty = (1 - sp.norm.cdf(numerator / denominator))  # survival function (1-cdf)

                else:
                    numerator1 = thresholds[ndx] - k[mask]
                    numerator2 = thresholds[ndx - 1] - k[mask]
                    probabilty = (sp.norm.cdf(numerator1 / denominator)
                                  - sp.norm.cdf(numerator2 / denominator))

                log_likelihood -= np.log(probabilty).sum()

            return log_likelihood

        rho = fminbound(_min_func, -.99, .99)

        # Likelihood ratio test
        log_likelihood_rho = _min_func(rho)
        log_likelihood_zero = _min_func(0.0)
        likelihood_ratio = -2 * (log_likelihood_rho - log_likelihood_zero)
        p_value = sp.chi2.sf(likelihood_ratio, 1)

        return rho, likelihood_ratio, p_value

    def biserial(self, df, col_name, num_cls):
        # print(df)
        target_col_name = 'class'
        feature_target_corr = {}
        if num_cls == 2:
            for col in col_name:
                if target_col_name != col:
                    rho, p = sp.pointbiserialr(df[col], df[target_col_name])
                    feature_target_corr[col + '_' + target_col_name] = rho
            print("biserial Done")
        else:
            for col in col_name:
                if target_col_name != col:
                    corr, likelihood, p = self.polyserial_correlation(df[col], df[target_col_name])
                    feature_target_corr[col + '_' + target_col_name] = corr

            print("polyserial Done")
        return feature_target_corr

    def score(self, human, X_C, X_Y, data, label, col_name):
        avg = np.zeros(human.shape[0])
        for i in range(human.shape[0]):
            score_sum = 0
            xc_sum = 0
            new_df = human[i, :]
            indx = np.where(new_df == 1)
            indx = indx[0]
            for x in range(self.gene_number - 1):
                idx = indx[x]
                x_c = X_C[col_name[idx] + '_class']

                if self.abs == 'XC':
                    xc_sum += float(x_c)
                elif self.abs == 'XCXY':
                    xc_sum += float(x_c)
                else:
                    xc_sum += float(abs(x_c))
                for a in range(x + 1, self.gene_number):
                    idx2 = indx[a]

                    cor = X_Y.iloc[idx, idx2]
                    # dataset = pd.DataFrame(data)
                    # x = dataset.iloc[:, idx]
                    # x = np.array(x)
                    #
                    # y = dataset.iloc[:, idx2]
                    # y = np.array(y)
                    #
                    # cor = np.corrcoef(x, y)[1][0]

                    # if cor < 0:
                    # print("xy_cor   = ", cor)
                    if self.abs == 'XY':
                        score_sum += cor
                    elif self.abs == 'XCXY':
                        score_sum += cor
                    else:
                        score_sum += abs(cor)

            # upper = abs(self.gene_number * (xc_sum / self.gene_number))
            # lower = (self.gene_number * (self.gene_number - 1)) * ((
            #             abs(score_sum / ((self.gene_number * (self.gene_number - 1))/2)))*0.5)

            upper = self.gene_number * (xc_sum / self.gene_number)
            lower = (self.gene_number * (self.gene_number - 1)) * (
                    score_sum / ((self.gene_number * (self.gene_number - 1)) / 2))

            xc = xc_sum / self.gene_number
            # print("X_C = ", xc)
            xy = score_sum / ((self.gene_number * (self.gene_number - 1))/2)
            if self.equation == 'mRMR':
            # avg[i] = abs(xc) + (1 / (self.gene_number * (self.gene_number - 1))) * abs(xy)  # upper / math.sqrt(self.gene_number + lower)
                avg[i] = xc - self.weight * xy  # upper / math.sqrt(self.gene_number + lower) #abs(xc) - self.weight * abs(xy)
            #     avg[i] = xc_sum - self.weight * score_sum #avg[i] = xc_sum / score_sum
            # print("average score: " + str(avg[i]))
            elif self.equation == 'noabsmRMR':
                avg[i] = xc_sum - self.weight * score_sum
            elif self.equation == 'noabsdiv':
                avg[i] = xc_sum / score_sum
            elif self.equation == 'CFS':
                avg[i] = upper / math.sqrt(self.gene_number + lower)  # upper / math.sqrt(self.gene_number + lower) #abs(xc) - self.weight * abs(xy)

        return avg

    def selection(self, avg, selection):
        a, b = np.random.choice(len(avg), 2)

        while a in selection:
            a = np.random.choice(len(avg), 1)
            a = a.tolist()
            a = a[0]
            if a not in selection:
                break
        while b in selection:
            b = np.random.choice(len(avg), 1)
            b = b.tolist()
            b = b[0]
            if b not in selection:
                break
        while a == b:
            b = np.random.choice(len(avg), 1)
            b = b.tolist()
            b = b[0]
            if a != b:
                break
        selection.append(a)
        selection.append(b)

        return a, b, selection

    def crossover(self, human, a, b):
        # print("-----crossover------")
        parent1 = human[a]
        parent2 = human[b]

        point = random.randint(0, self.col)

        off1 = np.concatenate((parent1[:point], parent2[point:]), axis=None)
        off2 = np.concatenate((parent2[:point], parent1[point:]), axis=None)

        return off1, off2

    def mutation(self, offspring):
        # print("-----mutation------")
        off1_idx = np.where(offspring == 1.0)[0]
        off0_idx = np.where(offspring == 0.0)[0]

        if len(off1_idx) == 0:
            pass
        else:
            off1_idx = list(off1_idx)
            select_mutate_point0 = random.choices(off1_idx, k=1)
            offspring = offspring.flatten()

            offspring[select_mutate_point0] = 0
        if len(off0_idx) == 0:
            pass
        else:
            off0_idx = list(off0_idx)
            select_mutate_point1 = random.choices(off0_idx, k=1)
            offspring = offspring.flatten()

            offspring[select_mutate_point1] = 1

        return offspring

    def evolve(self):
        print("-----evolve------")
        df, data, label, col_name, num_cls = extract_file(self.filename)
        # print(df, col_name)
        human = self.initialization()

        if self.method == 'Corr':
            file_path = 'XC_correlation_matrix_' + self.filename + '.json'

            if os.path.exists('../dataset/'+file_path):
                with open('../dataset/' + file_path, 'r') as file:
                    X_C = json.load(file)
            else:
                X_C = self.biserial(df, col_name, num_cls)
                with open('../dataset/' + file_path, 'w') as file:
                    json.dump(X_C, file)

            X_Y = data.corr(method='pearson')
            # file_path2 = 'correlation_matrix_' + self.filename + '.csv'
            # if os.path.exists('../dataset/'+file_path2):
            #     X_Y = dd.read_csv('../dataset/correlation_matrix_' + str(self.filename) + '.csv')
            #     print(X_Y)
            #     # X_Y = pd.DataFrame(X_Y)
            #     X_Y = X_Y.set_index(X_Y['Unnamed: 0'])
            #     print(X_Y)
            #     print(X_Y.head())
            # else:
            #     X_Y = data.corr(method='pearson')
            #     X_Y.to_csv('../dataset/correlation_matrix_' + str(self.filename) + '.csv', index=True)
            #     X_Y = dd.read_csv('../dataset/correlation_matrix_' + str(self.filename) + '.csv', index_col=0)
        else:
            file_path = 'XC_MI_matrix_' + self.filename + '.json'

            if os.path.exists('../dataset/'+file_path):
                with open('../dataset/' + file_path, 'r') as file:
                    X_C = json.load(file)
                # print(X_C)
            else:
                target_col_name = 'class'
                X_C = {}
                for i in range(data.shape[1]):
                    MI = mutual_info_classif(data.iloc[:, i].values.reshape(-1,1), label)
                    X_C[str(col_name[i]) + '_' + target_col_name] = float(MI[0])

                with open('../dataset/' + file_path, 'w') as file:
                    json.dump(X_C, file)

            file_path2 = 'MI_matrix_' + self.filename + '.csv'
            if os.path.exists('../dataset/'+file_path2):
                X_Y = pd.read_csv('../dataset/MI_matrix_' + str(self.filename) + '.csv', index_col=0)
                # print(X_Y)
                # print(ddddd)
            else:
                mi_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
                for i in range(0,data.shape[1]-1):
                    for j in range(i+1,data.shape[1]):
                        MI = mutual_info_regression(data.iloc[:, i].values.reshape(-1, 1), data.iloc[:, j].values)
                        mi_matrix.loc[data.columns[i], data.columns[j]] = float(MI[0])
                mi_matrix.to_csv('../dataset/MI_matrix_'+ str(self.filename) + '.csv', index=True)
                X_Y = pd.read_csv('../dataset/MI_matrix_' + str(self.filename) + '.csv', index_col=0)
        # print(human, X_C, X_Y, data, label, col_name)
        avg = self.score(human, X_C, X_Y, data, label, col_name)

        rank = avg.argsort()[::-1]
        avg_ = avg[rank]
        new_human = human[rank]
        threshold_GA = []
        for i in range(self.generation_num):
            print("-----" + str(i + 1) + " generation------")
            selection = []
            for j in range(self.human_number // 2):
                a, b, selection = self.selection(avg_, selection)
                crossover_rand = random.random()
                if crossover_rand <= 0.9:
                    off1, off2 = self.crossover(new_human, a, b)
                    mutate_rand = random.random()
                    if mutate_rand <= 0.05:
                        off1 = self.mutation(off1)
                        off2 = self.mutation(off2)
                    off1 = self.oneDetector(off1)
                    off2 = self.oneDetector(off2)
                else:
                    off1 = new_human[a]
                    off2 = new_human[b]
                off1 = off1.reshape(1, -1)
                off2 = off2.reshape(1, -1)
                new_human = np.concatenate((new_human, off1), axis=0)
                new_human = np.concatenate((new_human, off2), axis=0)
            parent = new_human[:self.human_number]
            offspring = new_human[self.human_number:]
            parent_avg = self.score(parent, X_C, X_Y, data, label, col_name)
            offspring_avg = self.score(offspring, X_C, X_Y, data, label, col_name)

            parent_rank = parent_avg.argsort()[::-1]
            offspring_rank = offspring_avg.argsort()[::-1]

            parent = parent[parent_rank]
            offspring = offspring[offspring_rank]
            parent_avg = parent_avg[parent_rank]
            offspring_avg = offspring_avg[offspring_rank]

            elite_rate = 0.2
            parent = parent[:int(self.human_number * elite_rate)]
            offspring = offspring[:int(self.human_number * (1 - elite_rate))]
            new_human = np.concatenate((parent, offspring), axis=0)

            avg_ = np.concatenate((parent_avg[:int(self.human_number * elite_rate)],
                                   offspring_avg[:int(self.human_number * (1 - elite_rate))]), axis=0)
            # avg = avg[rank]
            # avg = avg[:self.human_number]
            print("no.1 corr value = ", avg_.max())
            threshold_GA.append(avg_.max())
        self.drawGA(threshold_GA)

        self.savescv(new_human, df, label, col_name)
        print("GA Done")

    def oneDetector(self, baby):
        oneDetector = np.where(baby == 1)
        # print(oneDetector)
        oneDetector = oneDetector[0]
        if len(oneDetector) < self.gene_number:
            for a in range(self.gene_number - len(oneDetector)):
                zerotoone = np.where(baby == 0)
                zerotoone = zerotoone[0]
                rand_num = random.choice(zerotoone)

                baby[rand_num] = 1
        elif len(oneDetector) > self.gene_number:
            for a in range(len(oneDetector) - self.gene_number):
                onetozero = np.where(baby == 1)
                onetozero = onetozero[0]
                rand_num = random.choice(onetozero)
                baby[rand_num] = 0
        elif len(oneDetector) == self.gene_number:
            return baby
        return baby

    def savescv(self, new_human, df, label, col_name):
        f = open(
            "../result/After_GA/GA_" + str(self.filename) + "_" + str(self.human_number) + "_" + str(
                self.gene_number) + "_" + str(self.generation_num) + "_" + str(self.weight) + "_" + self.equation + "_" + self.abs + "_" + self.method + ".csv",
            'w')
        micro_array = []
        final_human = new_human[0]
        for a in range(self.col):
            if final_human[a] == 1:
                data = df.iloc[:, a]
                f.write(str(col_name[a]) + '\t')
                micro_array.append(df.columns[a])
                for n in range(self.row):
                    f.write(str(data[n]))
                    f.write('\t')

                f.write('\r\n')

        f.write("class" + '\t')
        for n in range(self.row):
            f.write(str(label[n]) + '\t')
        f.close()

    def drawGA(self, value):
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.plot(value)
        plt.savefig(
            '../result/GA_convergence/GA_convergence_' + str(self.filename) + '_GA_' + str(
                self.human_number) + "_" + str(self.gene_number) + "_" + str(
                self.generation_num) + "_" + str(self.weight) + "_" + self.equation + "_" + self.abs + "_" + self.method + '.png')