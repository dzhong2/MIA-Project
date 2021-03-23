# MIA-Project

## Datasets:
There are two folders containing the datasets we need, for now we only use Adult, Broward and Hosppital datasets:
1) In the "datasets" folder, we have the data without one hot transfer. This data can be used to train models like Decision tree or random forest.
   Datasets | Senesitive attribute index | Sensitive attribute value | Explaination 
   ------------ | ------------- | ------------- | ------------- 
   Adult | 9 | {0*,1} | {Female*, Male}
   Adult | 8 | 0* or others | White* or others
   Broward | 1 | {0, 1*} | {Male, Female*}
   Broward | 0 | 1* or others | Black* or others
   Hospital | 1 | {0, 1*} | {Male, Female*}
   Hospital | 2 | 1* or others | Black* or others
   
   The attribute value with * mark is the majority. Before feeding the data into models, the sensitive attribute need to be trasfered to binary.
   The categorical attributes are:
   Datasets | categorical attribute index 
   ------------ | ------------- 
   Adult | 1, 3, 5, 6, 7, 8, 9, 13 
   Broward | 0, 1 
   Hospital | 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
   
3) In the "One_hot" folder we have data that is preprocessed with one hot transfer. This one is needed for all the neural networks
   Datasets | Senesitive attribute index | Sensitive attribute value | Explaination 
   ------------ | ------------- | ------------- | ------------- 
   Adult | 0 | {0*,1} | {Female*, Male}
   Adult | 1 | {0*,1} | {White*, Black}
   Broward | 0 | {0*,1} | {Female*, Male}
   Broward | 1 | {0*,1} | {Black*, White}
   Hospital | 0 | {0*, 1} | {Female, Male}
   Hospital | 1 | {0*,1} | {Black*, White}
   
## Use the Auto_run.py:
The script Auto_run.py includes the three parts: Target experiments, MIA experiments and PLD measure. To run this code, you have options for following parameters:
1. -rep

   This parameter means the repeating time for target experiments. The defalt value is 5.
2. --file_list

   This parameter means the datasets you want to run experiments on. The defalt value is [0,1,2], which means all the datasets will be run on.
   
## Expected Output
For each step, these results are expected:
1. Target experiment:

There should be results in /result_by_step/[dataset_name]/time=[i]/target_result.csv

2. MIA experiment

There should be results in /result_by_step/[dataset_name]/time=[i]/attack_result_[i].csv

3. PLD Calculation

There should be results in /MIA_result/All_PLD_negative.csv and /MIA_result/All_PLD_positive.csv
   
## Use your own model.
To use your own model, please follow these instructions:
1. If you are using neural network, please use the one-hot transfered data.
2. Please make sure your model has following functions:

   1) model.fit(X, y)
   2) model.predict_proba(X)
3. Then you can replace the code in Generate_target_result.py. You can put it in line 50 to line 55
