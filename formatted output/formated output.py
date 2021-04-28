import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Tools import plot_prob_distribution, get_dsitribution, KLD

def find_all_results(root, result_type="target_result.csv"):
    '''output: all the result file name and their paths'''
    file_list = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if result_type in name:
                file_list.append(os.path.join(path, name).replace("\\", "/"))
    return file_list

def get_accuracy(files):
    '''Get train/test accuracy from each file.
    output: three list of accuracy: train, test, gap
            list of nm'''
    pct_list = []
    train_acc_list = []
    test_acc_list = []
    gap_list = []
    dataset_list = []
    for file in files:
        df = pd.read_csv(file, header=None)
        result = np.array(df)
        train_result = result[result[:, -1] == 1, :]
        train_pred = np.argmax(train_result[:, 1:3], axis=1)
        train_acc = sum(train_pred == train_result[:, 0]) / len(train_pred)

        test_result = result[result[:, -1] == 0, :]
        test_pred = np.argmax(test_result[:, 1:3], axis=1)
        test_acc = sum(test_pred == test_result[:, 0]) / len(test_pred)
        if "dropout_0_percent" in file:
            pct = "no-drop-out"
        else:
            pct = "0"
        dataset_list.append(file.split('/')[2])
        pct_list.append(pct)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc),
        gap_list.append((train_acc - test_acc))

    tmp_df = np.array([dataset_list, train_acc_list, test_acc_list, gap_list])
    df = pd.DataFrame(tmp_df.T, columns=['dataset', 'train', 'test', 'gap'])
    df = df.astype({'train': 'float',
                    'test': 'float',
                    'gap': 'float'})
    df_avg = df.groupby('dataset').mean()

if __name__ == "__main__":
    # Target model performance without defense
    files = find_all_results("../result_by_step")
    get_accuracy(files)
