import pandas as pd
import numpy as np

RACELESS = True
RACE_COLS = [ "African_American", "Asian", "Hispanic", "Native_American" ]

BUCKET_PRIORS = True
FIXED_BUCKETS = False
fixed_size = 1
EXPONENTIAL_BUCKETS = True
expo_size = 1.8

def load_data(fname):
    return pd.read_csv(fname)


def get_split_binary_data(fname="propublica_data_for_fairml.csv", n=None):
    dataframe = load_data(fname)
    attributes_to_drop = ["score_factor", "Other"] # ["Other"]
    dataframe = dataframe.drop(attributes_to_drop, axis=1).copy()
    max_num_priors = np.max(dataframe["Number_of_Priors"].to_numpy())
    print("Max number of priors is:", max_num_priors)
    min_num_priors = np.min(dataframe["Number_of_Priors"].to_numpy())
    print("Min number of priors is:", min_num_priors)
    if BUCKET_PRIORS :
        num_criminals = dataframe.shape[0]
        bucket_min = 0
        bucket_max = 1
        while bucket_min <= max_num_priors :
            list = [ 1 if bucket_min <= dataframe["Number_of_Priors"][j] and dataframe["Number_of_Priors"][j] < bucket_max else 0 for j in range(num_criminals) ]
            dataframe[str(bucket_min) + "<=Priors<" + str(bucket_max)] = list
            if FIXED_BUCKETS :
                bucket_min = bucket_max
                bucket_max += fixed_size
            elif EXPONENTIAL_BUCKETS :
                prev_range = bucket_max - bucket_min
                new_range = prev_range * expo_size
                bucket_min = bucket_max
                bucket_max += new_range
            else : exit(1)
        dataframe = dataframe.drop("Number_of_Priors", axis=1).copy()
    positiveDF = dataframe[dataframe["Two_yr_Recidivism"] == 1].copy()
    negativeDF = dataframe[dataframe["Two_yr_Recidivism"] == 0].copy()
    if n!=None: class_size=n
    else: class_size = 2 * positiveDF.shape[0] // 3
    X_train = (
        pd.concat([positiveDF[:class_size], negativeDF[:class_size]])
        .reset_index(drop=True)
        .copy()
    )
    X_test = (
        pd.concat([positiveDF[class_size:], negativeDF[class_size:]])
        .reset_index(drop=True)
        .copy()
    )
    Y_train = X_train["Two_yr_Recidivism"].values.copy()
    Y_test = X_test["Two_yr_Recidivism"].values.copy()
    X_train = X_train.drop("Two_yr_Recidivism", axis=1)
    X_test = X_test.drop("Two_yr_Recidivism", axis=1)
    if RACELESS :
        for race in RACE_COLS :      
            X_train = X_train.drop(race, axis=1)
            X_test = X_test.drop(race, axis=1)
    column_names = X_test.columns
    features = { column_names[i] : i for i in range(len(column_names)) }
    # print("Number of correct COMPAS predictions on the test set:", np.sum(X_test["score_factor"] == Y_test))
    # print("COMPAS Accuracy:", np.sum(X_test["score_factor"] == Y_test) / X_test.shape[0])
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    return (features, X_train, Y_train, X_test, Y_test)