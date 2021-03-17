# MIA-Project

## Datasets:
There are two folders containing the datasets we need, for now we only use Adult, Broward and Hosppital datasets:
1) in the "datasets" folder, we have the data without one hot transfer. This data can be used to train models like Decision tree or random forest.
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
   Broward | 1, 3, 5, 6, 7, 8, 9, 13 
   Hospital | 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
   
3) In the "One_hot" folder we have data that is preprocessed with one hot transfer. This one is needed for all the neural networks
   Datasets | Senesitive attribute index | Sensitive attribute value | Explaination 
   ------------ | ------------- | ------------- | ------------- 
   Adult | 9 | {0*,1} | {Female*, Male}
   Adult | 8 | 0* or others | White* or others
   Broward | 1 | {0, 1*} | {Male, Female*}
   Broward | 0 | 1* or others | Black* or others
   Hospital | 1 | {0, 1*} | {Male, Female*}
   Hospital | 2 | 1* or others | Black* or others
