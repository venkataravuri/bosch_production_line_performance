import pandas as pd

from global_config import config


def load_raw_numeric_data():
    df_train_numeric = pd.read_csv(config.raw_data_folder + "/" + config.raw_train_numeric_file, index_col='Id')
    print("Raw numeric training data info:\n")
    print(df_train_numeric.info())

    df_train_numeric.fillna(0, inplace=True)
    df_train_numeric_sparse = df_train_numeric.to_sparse(fill_value=0)
    print("Convert to numeric training data into sparse dataframe:\n")
    print(df_train_numeric_sparse.info())

    #print("Training numeric data:\n")
    #print(df_train_numeric_sparse.head())

    return df_train_numeric_sparse


def save_numeric_data():
    df_train_numeric_sparse.to_pickle(config.processed_data_folder + "/" + config.processed_train_numeric_file)


if __name__ == "__main__":
    df_train_numeric_sparse = load_raw_numeric_data()

    save_numeric_data()



