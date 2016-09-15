import os

class GlobalConfig:
    def __init__(self):
        # Paths
        self.data_folder = os.path.dirname(__file__) + "/../data"
        self.raw_data_folder = self.data_folder + "/raw"
        self.processed_data_folder = self.data_folder + "/processed"
        self.interim_data_folder = self.data_folder + "/interim"
        self.external_data_folder = self.data_folder + "/external"

        # File
        self.processed_train_numeric_file = "train_numeric.pkl"
        self.raw_train_numeric_file = "train_numeric.csv"

# Initialize global config
config = GlobalConfig()