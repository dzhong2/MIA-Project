from NN_DP import attack_model
import pandas as pd
import numpy as np
from Tools import find_all_results
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf


def run_MIA(args, dataset_name = ""):
    #### You can change following dataset name and target result location ####
    if dataset_name == "":
        target_result_location = "result_by_step"
    else:
        target_result_location = "result_by_step/" + dataset_name
    max_time = args.rep[0]
    # Find all the target model results
    files = find_all_results(target_result_location)
    # A flag. If replace==1, then MIA will always generate new result and overwrite the old one (if there is one)
    replace = 0
    dataset_list = np.array(['Adult', 'Broward', 'Hospital'])[args.file_list]
    for file in files:
        if file.split('/')[1] not in dataset_list:
            print(file.split('/')[1], "dataset is not in the list")
            continue
        if int(file.split('time=')[1].split('/')[0]) >= max_time:
            print("Skip time =", file.split('time=')[1].split('/')[0], "because repeating number is", max_time)
            continue
        ## You can change this repeating time. Now it will run 2 MIA experiments on each target result ##
        for time in range(2):
            folder = file.split("/target_result.csv")[0]
            if os.path.exists(folder + "/attack_result_" + str(time)+".csv") and replace == 0:
                print("Skipped " + file + "time=" + str(time))
                continue
            else:
                print("Start working on " + file + "time=" + str(time))
            df = pd.read_csv(file, header=None)
            data = np.array(df)
            X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.5)
            tmp_all = []
            for c in [0, 1]:
                inds = X_train[:, 0] == c
                X_c = X_train[inds, 1:3]
                gender_c = X_train[inds, 3]
                race_c = X_train[inds, 4]
                ##### Please replace your own model with following code #####
                # Your model should have function: #
                # model.fit, model.predict_proba #
                model = attack_model(num_epoch=15000,
                                     learning_rate=1e-5,  # 5e-5 works for both
                                     batch_size=4000,
                                     verbose=2)
                ##### Please replace the model above #####
                y_c = y_train[inds]
                model.fit(X_c, y_c)
                x_prob = model.predict_proba(X_c)
                All_data_c = np.hstack((y_c.reshape(-1, 1),
                                        x_prob,
                                        gender_c.reshape(-1, 1),
                                        race_c.reshape(-1, 1),
                                        X_train[inds, 0].reshape(-1, 1)))
                tmp_all.append(All_data_c)
            All_Data = np.vstack((tmp_all[0], tmp_all[1]))
            final_df = pd.DataFrame(All_Data, index=None)
            final_df.to_csv(folder + "/attack_result_" + str(time)+".csv", header=False, index=False)



