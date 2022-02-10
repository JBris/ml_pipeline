##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys
import pandas as pd
import tempfile

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import get_config
from pipeline_lib.data import Data

##########################################################################################################
### Constants
##########################################################################################################

# Config
PROJECT_NAME = "kiwi_datasets"
CONFIG = get_config(base_dir)
DATA = Data()

# Random
RANDOM_STATE = CONFIG.get("random_seed") 

##########################################################################################################
### Pipeline
##########################################################################################################

def _add_interactions(path: str):
    df = DATA.read_csv(f"data/{path}.csv")
    df["x_calyx_MA_rel"] = df["x"] * df["calyx_MA_rel"]
    df["x_maMAratio"] = df["x"] * df["maMAratio"]
    df["height_calyx_ma"] = df["height"] * df["calyx_ma"]
    df["perimeter_coff"] = df["perimeter"] * df["coff"]
    df.to_csv(f"data/{path}_extended.csv", index = False)
    
def main() -> None:
    _add_interactions("hw_data_4217")
    _add_interactions("hw_data_7154")
    _add_interactions("ga_data_4217")
    _add_interactions("ga_data_7154")
    _add_interactions("ga_data_combined")
    _add_interactions("hw_data_combined")
        
if __name__ == "__main__":
    main()
     