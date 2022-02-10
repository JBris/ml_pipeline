##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys
import tempfile

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import get_config
from pipeline_lib.data import Data, join_path

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

def main() -> None:
    df = DATA.read_csv("data/hw_data_7154.csv")
    df["x_calyx_MA_rel"] = df["x"] * df["calyx_MA_rel"]
    df["x_maMAratio"] = df["x"] * df["maMAratio"]
    df["height_calyx_ma"] = df["height"] * df["calyx_ma"]
    df["perimeter_coff"] = df["perimeter"] * df["coff"]
    df.to_csv("data/hw_data_7154_extended.csv")
        
if __name__ == "__main__":
    main()
     