"""
Machine Learning Pipeline Data

A library for data loading and manipulation within the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

# External
import os
import pandas as pd


##########################################################################################################
### Library  
##########################################################################################################

def join_path(p1: str, p2: str) -> str:
    return os.path.join(p1, p2)

class Data:
    def read_csv(self, name: str, **kwargs) -> pd.DataFrame:
        """
        Wrapper method to load a csv file.

        Parameters
        --------------
        name: str
            The file name.
        **kwargs
            Additional arguments.

        Returns
        ---------
        df: DataFrame
            The dataframe.
        """
        return pd.read_csv(name, **kwargs)

    def train_test_split(self, df: pd.DataFrame, frac: float = 0.9, random_state: int = None) -> tuple:
        """
        Perform a train-test split.

        Parameters
        --------------
        df: DataFrame
            The dataframe.
        frac: float
            The fraction of training data.
        random_state: int
            The random seed.

        Returns
        ---------
        dfs: (DataFrame, DataFrame)
            The training and testing dataframes.
        """
        data = df.sample(frac = frac, random_state = random_state)
        data_unseen = df.drop(data.index)
        data.reset_index(drop = True, inplace = True)
        data_unseen.reset_index(drop = True, inplace = True)
        return data, data_unseen

    def get_filename(self, config) -> str:
        """
        Return the file name for the input data.

        Parameters
        --------------
        config: Config
            The configuration object.

        Returns
        ---------
        file_name: str
            The file name for the input data.
        """
        base_dir = config.get("base_dir", False)
        file_path = config.get("file_path")

        if base_dir is None:
            raise Exception(f"Directory not defined error: {base_dir}")
            
        if file_path is None:
            raise Exception(f"File path not defined error: {file_path}")

        file_name = join_path(base_dir, file_path)
        return file_name

    def query(self, config, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wrapper method for performing dataframe queries.

        Parameters
        --------------
        config: Config
            The configuration object.
        df: DataFrame
            The dataframe.

        Returns
        ---------
        df: DataFrame
            The queried dataframe.
        """
        df_query = config.get("df_query")
        if df_query is None:
            return df
        return df.query(df_query).reset_index()
        