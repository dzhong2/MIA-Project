'''
Use one hot datasets to train N target models, save raw_data+prob+train_test_label
to target_result/dataset_name/nm=?/time=N/
'''



import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
from DT_DP import node
from Tools import race_map

def run(args):
    FileNames = ['datasets/Adult.csv',
                 'datasets/Compas.csv',
                 'datasets/Broward.csv',
                 'datasets/Hospital.csv'
                 ]
    rep = args.rep[0]
    sensitive_inds = [[9, 8], [1, 2], [1, 0], [1, 2]]
    protect_races = [0, 1, 1, 1]
    protect_genders = [0, 0, 1, 1]
    for file_ind in args.file_list:
        file = FileNames[file_ind]
        gender_ind = sensitive_inds[file_ind][0]
        race_ind = sensitive_inds[file_ind][1]
        for time in range(rep):
            df = pd.read_csv(file, header=None)
            data = np.array(df)
            data[:, gender_ind] = race_map(data[:, gender_ind], protect_genders[file_ind])
            data[:, race_ind] = race_map(data[:, race_ind], protect_races[file_ind])
            X_train, X_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1], test_size=0.5)
            gender_train = X_train[:, gender_ind]
            race_train = X_train[:, race_ind]
            gender_test = X_test[:, gender_ind]
            race_test = X_test[:, race_ind]
            folder = "../result_by_step/" + file.split("datasets/")[1].split(".csv")[0] + \
                                 '/time=' + str(time+10)
            if os.path.exists(folder + "/target_result.csv"):
                print("Skipped" + folder)
                continue
            else:
                print("Start Working on " + folder)
                '''Please replace following code with your own model'''
                model = node(max_depth=8)
                model.fit(X_train, y_train, dp=0, ep=5.0)
                save_result(model, X_train, X_test, y_train, y_test, gender_train, gender_test, race_train,
                                    race_test, folder)

def save_result(model, X_train, X_test, y_train, y_test, gender_train, gender_test, race_train, race_test, folder):
    if os.path.exists(folder + "/target_result.csv"):
        print("Skipped " + folder)
        return 0
    if not os.path.exists(folder):
        os.makedirs(folder)
    prob_train = model.predict_proba(X_train)
    prob_test = model.predict_proba(X_test)
    Train_Data = np.hstack((y_train.reshape(-1, 1),
                            prob_train,
                            gender_train.reshape(-1, 1),
                            race_train.reshape(-1, 1),
                            np.ones([len(y_train), 1])))
    Test_Data = np.hstack((y_test.reshape(-1, 1),
                           prob_test,
                           gender_test.reshape(-1, 1),
                           race_test.reshape(-1, 1),
                           np.zeros([len(y_test), 1])))
    AllData = np.vstack((Train_Data, Test_Data))

    final_df = pd.DataFrame(AllData, index=None)
    final_df.to_csv(folder + '/target_result.csv', header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-rep', type=int, nargs='+', default=4, help='number of repeating experimants')
    parser.add_argument('--file_list', type=int, nargs='+', default=[3], help='which dataset')
    args = parser.parse_args()
    run(args)
