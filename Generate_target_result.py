'''
Use one hot datasets to train N target models, save raw_data+prob+train_test_label
to target_result/dataset_name/nm=?/time=N/
'''



import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
from NN_DP import target_model
from Tools import race_map

def run_target(args):
    FileNames = ['One_hot/ohAdult.csv',
                 'One_hot/ohBroward.csv',
                 'One_hot/ohHospital.csv'
                 ]
    lr_list = [1e-3, 1e-3, 3.5e-3]
    batch_ratio = [80, 80, 100]
    epoch_ratio = [4, 80, 5]
    rep = args.rep[0]
    for file_ind in args.file_list:
        file = FileNames[file_ind]
        lr = lr_list[file_ind]
        epoch_rate = epoch_ratio[file_ind]
        for time in range(rep):
            folder = "result_by_step/" + file.split("oh")[1].split(".csv")[0] + \
                     '/time=' + str(time)
            if os.path.exists(folder + "/target_result.csv"):
                print("Skipped" + folder)
                continue
            else:
                print("Start Working on " + folder)
            df = pd.read_csv(file, header=None)
            data = np.array(df)
            X_train, X_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1], test_size=0.5)
            gender_train = X_train[:, 0]
            race_train = X_train[:, 1]
            gender_test = X_test[:, 0]
            race_test = X_test[:, 1]
            data_size = len(y_train)
            batch_size = round(data_size / batch_ratio[file_ind]) - 1
            epoch = int(data_size * epoch_rate / 200)
            delete_data = data_size % batch_size
            X_train_d = X_train[0:(data_size - delete_data), :]
            y_train_d = y_train[0:(data_size - delete_data)]
            model = target_model(num_epoch=epoch,
                                 num_microbatches=batch_size,
                                 learning_rate=lr,
                                 data_size=data_size,
                                 verbos=1)
            model.fit(X_train_d, y_train_d,epoch=epoch)
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
    parser.add_argument('-rep', type=int, nargs='+', default=5, help='number of repeating experimants')
    parser.add_argument('--file_list', type=int, nargs='+', default=[0, 1, 2], help='which dataset')
    args = parser.parse_args()
    run_target(args)
