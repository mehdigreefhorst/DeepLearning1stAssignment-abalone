import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

class AbalonDataset(Dataset):
    def __init__(self, data_df, train_dataset: "AbalonDataset" = None):
        """if train_dataset is provided, then it is assumed that this dataset must be the test dataset. We then access the mean and std of the provided dataset"""
        self.data = data_df
        #standardized_data = 
        if train_dataset is None:
            # We only want to split train and val set when the provided dataset is the train dataset so not the test dataset 
            # we only want to calculate the mean and std based on the train set
            self.mean = self.get_mean(self.data)
            self.std = self.get_std(self.data)
        else:
            self.mean = train_dataset.mean
            self.std = train_dataset.std
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """we first take the row, then we pop Rings and Sex. Then we make a sex tensor which is a category, 
        then we make tensors of the numerical columns. We combine the sex and numerical into features. We also make a tensor of the dependent variable: Rings"""
        row = self.preprocess_data(self.data.iloc[index])
        target_col = None
        if "Rings" in row:
            target_col = row.pop("Rings")
        sex_feature = row.pop("Sex")
        # we make the sex tensor separate because it must be onehot encoded
        sex_tensor = torch.nn.functional.one_hot(torch.tensor(sex_feature, dtype=torch.long), num_classes=3).to(torch.float32)
        numerical_tensor = torch.tensor(row, dtype=torch.float32)
        # standardize the numerical features
        standardized_numerical_tensor = (numerical_tensor - self.mean) / (4* self.std)
        features = torch.concat([sex_tensor, standardized_numerical_tensor])

        # when target_col is none, the data is training or validation data
        if target_col is not None:
            target = torch.tensor(target_col, dtype=torch.float32).view(1)
        # in case the dataset is the test dataset where there is no Rings column, we need to create an empty target
        else:
            target = torch.tensor([0])
        return features, target

    @staticmethod
    def preprocess_data(row):
        """preproces data of sex to numerical values"""
        sex_map = {"M": 0, "F": 1, "I": 2}
        row = row.copy()
        row["Sex"] = sex_map[row["Sex"]]
        return row
    
    @staticmethod
    def get_mean(df):
        data = df.drop(["Sex", "Rings"], axis=1)
        return torch.tensor(data.mean().values, dtype=torch.float32)
    
    @staticmethod
    def get_std(df):
        data = df.drop(["Sex", "Rings"], axis=1)
        return torch.tensor(data.std().values, dtype=torch.float32)
    
    @classmethod
    def create_train_test_val(cls, csv_file_name, train_dataset: "AbalonDataset" = None):
        """class method to create a train test and val object of the AbalonDataset"""
        data = pd.read_csv(f"data/{csv_file_name}")
        if train_dataset is None:
            train_df, val_df = train_test_split(data, test_size = 0.2)
            trainAbalone = cls(train_df)
            valAbalone = cls(val_df, trainAbalone)
            return trainAbalone, valAbalone
        else:
            return cls(data, train_dataset)