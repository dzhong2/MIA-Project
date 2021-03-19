from sklearn import metrics
import numpy as np
import os
import matplotlib.pyplot as plt

def score_calculate(predict_y, y):
    if len(y) >1 and sum(predict_y == 1)>0:
        prec = metrics.precision_score(y, predict_y)
        recall = metrics.recall_score(y, predict_y)
        f1 = metrics.f1_score(y, predict_y)
    elif sum(abs(predict_y - y)) == 0:
        prec = 1
        recall = 1
        f1 = 1
    else:
        prec = 0
        recall = 0
        f1 = 0
    return prec, recall, f1

def pld_calculate(predict_y, y, gender, race):
    # 1 is protected
    TP_race1 = sum((race == 1) * (y == 1) * (predict_y == 1))
    TN_race1 = sum((race == 1) * (y == 0) * (predict_y == 0))
    FP_race1 = sum((race == 1) * (y == 0) * (predict_y == 1))
    FN_race1 = sum((race == 1) * (y == 1) * (predict_y == 0))

    TP_race0 = sum((race == 0) * (y == 1) * (predict_y == 1))
    TN_race0 = sum((race == 0) * (y == 0) * (predict_y == 0))
    FP_race0 = sum((race == 0) * (y == 0) * (predict_y == 1))
    FN_race0 = sum((race == 0) * (y == 1) * (predict_y == 0))
    if TP_race0==0 or TP_race1==0 or TN_race0==0 or TN_race1==0:
        print("Pause")

    pld_race_acc = (TP_race1 + TN_race1) / (TP_race1 + FP_race1 + FN_race1 + TN_race1) - \
                   (TP_race0 + TN_race0) / (TP_race0 + FP_race0 + FN_race0 + TN_race0)

    pld_race_prec = TP_race1 / (TP_race1 + FP_race1) - TP_race0 / (TP_race0 + FP_race0)

    pld_race_recall = TP_race1 / (TP_race1 + FN_race1) - TP_race0 / (TP_race0 + FN_race0)

    TP_gender1 = sum((gender == 1) * (y == 1) * (predict_y == 1))
    TN_gender1 = sum((gender == 1) * (y == 0) * (predict_y == 0))
    FP_gender1 = sum((gender == 1) * (y == 0) * (predict_y == 1))
    FN_gender1 = sum((gender == 1) * (y == 1) * (predict_y == 0))

    TP_gender0 = sum((gender == 0) * (y == 1) * (predict_y == 1))
    TN_gender0 = sum((gender == 0) * (y == 0) * (predict_y == 0))
    FP_gender0 = sum((gender == 0) * (y == 0) * (predict_y == 1))
    FN_gender0 = sum((gender == 0) * (y == 1) * (predict_y == 0))
    if TP_gender0==0 or TP_gender1==0 or TN_gender0==0 or TN_gender1==0:
        print("Pause")

    pld_gender_acc = (TP_gender1 + TN_gender1) / (TP_gender1 + FP_gender1 + FN_gender1 + TN_gender1) - \
                     (TP_gender0 + TN_gender0) / (TP_gender0 + FP_gender0 + FN_gender0 + TN_gender0)

    pld_gender_prec = TP_gender1 / (TP_gender1 + FP_gender1) - TP_gender0 / (TP_gender0 + FP_gender0)

    pld_gender_recall = TP_gender1 / (TP_gender1 + FN_gender1) - TP_gender0 / (TP_gender0 + FN_gender0)

    return pld_gender_acc, pld_gender_prec, pld_gender_recall, \
           pld_race_acc, pld_race_prec, pld_race_recall

def race_map(s, prot_value):
    result = np.ones(len(s))
    result[s != prot_value] = 0
    return result

def get_min_max(data):
    min = data.min(axis=0)[2:-1]
    max = data.max(axis=0)[2:-1]
    result = np.stack([min, max])
    return tuple(map(tuple, result))

def find_all_results(root_path):
    '''output: all the result file name and their paths'''
    file_list = []
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            if "target_result.csv" in name:
                file_list.append(os.path.join(path, name).replace("\\", "/"))
    return file_list

def plot_prob_distribution(prob_train, prob_test, scale, file_name):
    folder = file_name.split("target_result.csv")[0]

    bins = np.linspace(0, 1, num=int(1 // scale))
    plt.figure("Target Prob Distribution")
    plt.hist([prob_train[:, 0], prob_test[:, 0]], bins, label=['train', 'test'])
    plt.title(file_name)
    plt.xlabel("Prob_0")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(folder + 'Target_distribution.png')
    plt.close()

def get_dsitribution(probs, bin_size=0.05):
    probs = np.array(probs)
    bin_number = int(1/bin_size)
    count_all = len(probs)
    pdf = []
    for b_ind in range(bin_number):
        counti = ((probs >= b_ind*bin_size)*(probs <= (1+b_ind)*bin_size)).sum()
        pdfi = (counti+1)/(count_all + bin_number)
        pdf.append(pdfi)
    return pdf

def KLD(p, q):
    p = np.array(p)
    q = np.array(q)
    p = p/p.sum()
    q = q/q.sum()
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
