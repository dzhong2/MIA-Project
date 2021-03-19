import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from Tools import plot_prob_distribution
from sklearn import metrics


def find_all_results(data_name):
    '''output: all the result file name and paths'''
    file_list = []
    root_path = "result_by_step/"+data_name
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            if "attack_result" in name:
            #print(os.path.join(path, name))
                file_list.append(os.path.join(path, name))
    return file_list

def get_accuracy(files):
    '''Get train/test accuracy from each file.
    output: three list of accuracy: train, test, gap
            list of ep'''
    ep_list = []
    acc_list = []
    prec_list = []
    recall_list = []
    for file in files:
        df = pd.read_csv(file, header=None)
        result = np.array(df)
        labels = result[:, 0]
        pred = result[:, 1].round()
        acc = metrics.accuracy_score(labels, pred)
        prec = metrics.precision_score(labels, pred)
        recall = metrics.recall_score(labels, pred)
        if "ndp" in file:
            ep = "ndp"
        else:
            ep = file.split("ep=")[1].split("\\")[0]
        ep_list.append(ep)
        acc_list.append(acc)
        prec_list.append(prec),
        recall_list.append(recall)

    return ep_list, acc_list, prec_list, recall_list

def get_pld(data, ind=1, score="prec", mode=0):
    pred_sens = data[data[:, ind]==0, 1].round()
    pred_usens = data[data[:, ind]==1, 1].round()
    label_sens = data[data[:, ind]==0, 0]
    label_usens = data[data[:, ind]==1, 0]

    if score == "prec":
        prec_sens = metrics.precision_score(label_sens, pred_sens)
        prec_usens = metrics.precision_score(label_usens, pred_usens)
        pld = pld_Pr(prec_sens, prec_usens, mode)
    elif score =="acc":
        acc_sens = metrics.accuracy_score(label_sens, pred_sens)
        acc_usens = metrics.accuracy_score(label_usens, pred_usens)
        pld = pld_Pr(acc_sens, acc_usens, mode)
    else:
        recall_sens = metrics.recall_score(label_sens, pred_sens)
        recall_usens = metrics.recall_score(label_usens, pred_usens)
        pld = pld_Pr(recall_sens,recall_usens,mode)
    return pld

def pld_Pr(x1, x2, mode=0):
    if mode==0:
        return x1 - x2
    else:
        return max(x1/x2, x2/x1)

def disparity_measure(files, mode=0):
    ep_list = []

    pld_gender_prec = []
    pld_gender_recall = []
    pld_gender_acc = []

    pld_race_prec = []
    pld_race_recall = []
    pld_race_acc = []
    for file in files:
        df = pd.read_csv(file, header=None)
        result = np.array(df)

        pld_race_acc.append(get_pld(result, 3, "acc", mode))
        pld_race_recall.append(get_pld(result, 3, "recall", mode))
        pld_race_prec.append(get_pld(result, 3, "prec", mode))
        pld_gender_acc.append(get_pld(result, 2, "acc", mode))
        pld_gender_recall.append(get_pld(result, 2, "recall", mode))
        pld_gender_prec.append(get_pld(result, 2, "prec", mode))

        if "ndp" in file:
            ep = "ndp"
        else:
            ep = file.split("ep=")[1].split("\\")[0]
        ep_list.append(ep)

    return [ep_list,
            pld_gender_prec,
            pld_gender_recall,
            pld_gender_acc,
            pld_race_prec,
            pld_race_recall,
            pld_race_acc]

def plot_accuracy(tmp_df, data_name):
    '''Plot and save the results
    '''
    df = pd.DataFrame(tmp_df.T, columns=['ep', 'acc', 'prec', 'recall'])
    df['ep'][df['ep'] == 'ndp'] = 10
    df = df.astype("float")
    avg = df.groupby("ep").mean()
    avg.index = np.concatenate((avg.index.values[:-1], ['ndp']))
    plt.figure(data_name)

    plt.plot('acc', data=avg)
    plt.plot('prec', data=avg)
    plt.plot('recall', data=avg)
    plt.xlabel("Epsilon")
    plt.ylabel("Scores")
    plt.legend()
    folder = "MIA_result/MIA_score/"+data_name+"_MIA_score.png"
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(folder)
    plt.close()

def plot_pld(tmp_df, data_name):
    '''Plot and save the results
    '''
    df = pd.DataFrame(tmp_df.T, columns=['ep',
                                         'pld_gender_prec',
                                         'pld_gender_recall',
                                         'pld_gender_acc',
                                         'pld_race_prec',
                                         'pld_race_recall',
                                         'pld_race_acc'])
    df['ep'][df['ep'] == 'ndp'] = 10
    df = df.astype("float")
    pos, neg, sV = abs_symble(df, data_name)



def pld_reduce_perc(pld_df):
    reduce = -(pld_df - pld_df.loc[10]) / pld_df.loc[10]
    reduce_text = '('+(reduce*100).round(1).astype(str)+'%)'
    final_text = pld_df.round(3).astype(str)+' '+reduce_text
    return final_text

def abs_symble(df, data_name):
    values = df.set_index('ep').abs().groupby("ep").mean()
    symbles = ((df.set_index('ep') >= 0).groupby("ep").mean()>0.5)*2-1
    pos_counts = (df.set_index('ep') >= 0).groupby("ep").sum()
    pos_values = (df.set_index('ep')*(df.set_index('ep') >= 0)).groupby("ep").sum()/pos_counts
    neg_counts = (df.set_index('ep') <= 0).groupby("ep").sum()
    neg_values = (df.set_index('ep') * (df.set_index('ep') <= 0)).groupby("ep").sum() / neg_counts

    pos_text = pld_reduce_perc(pos_values)
    neg_text = pld_reduce_perc(neg_values)
    folder = "MIA_result/PLD/" + data_name
    pos_text.to_csv(folder + ".csv")
    neg_text.to_csv(folder + "neg_pld.csv")

    '''Following code are not used currently.'''
    symbles_pld = values*symbles
    s = values.index
    values = values.set_index([s, np.exp(s)])
    return pos_values, neg_values, symbles_pld

def target_distribution(files):
    for file in files:
        df = pd.read_csv(file, header=None)
        result = np.array(df)
        prob_train = result[result[:, 3] == 1, 1:3]
        prob_test = result[result[:, 3] == 0, 1:3]
        plot_prob_distribution(prob_train, prob_test, 0.01, file)



if __name__ == "__main__":
    mode = [0]
    dataset_name = "Adult" # Adult, Broward or Hospital
    result_location = "Adult"
    files = find_all_results(result_location)
    '''The middle results for all the attack models will be saved
     as result_by_step/[dataset_name]/ep=?/time=?/attack_result.csv'''
    if 0 in mode:
        '''In this mode, the code will get accuracy, precision and recall rate of attack experiment
        and save it into a figure.'''
        ep_list, acc_list, prec_list, recall_list = get_accuracy(files)
        tmp_df = np.array([ep_list, acc_list, prec_list, recall_list])
        plot_accuracy(tmp_df, dataset_name)
    if 1 in mode:
        '''In this mode, the code will measure the PLD of three scores on different sensitive attributes.'''
        tmp_df = np.array(disparity_measure(files, mode=1))
        plot_pld(tmp_df, dataset_name)
    if 2 in mode:
        '''In this mode, the code will measure the KL divergence between training and testing output. They
                will be measured on different groups (male vs female etc.)'''
        target_distribution(files)