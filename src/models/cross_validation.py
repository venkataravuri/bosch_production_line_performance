import numpy as np
import pandas as pd
import xgboost as xgb

from global_config import config


def load_train_data():
    df_train_numeric_sparse = pd.read_pickle(config.processed_data_folder + "/" + config.processed_train_numeric_file)
    print("loading of " + config.processed_data_folder + "/" + config.processed_train_numeric_file + " done.")
    print(df_train_numeric_sparse.info())
    return df_train_numeric_sparse


### load data in do training
df_train_numeric_sparse = load_train_data()
y = df_train_numeric_sparse['Response']
df_train_numeric_sparse.drop('Response', axis=1, inplace=True)
dtrain = xgb.DMatrix(df_train_numeric_sparse, label=y)
param = {'max_depth': 2, 'eta': 1, 'silent': 0, 'objective': 'binary:logistic'}
num_round = 2

print('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
xgb.cv(param, dtrain, num_round, nfold=3,
       metrics={'error'}, seed=0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

