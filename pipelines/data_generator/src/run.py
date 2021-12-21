##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse

import os, sys
sys.path.insert(0, os.path.abspath('../..'))

# Internal 
from pipeline_lib.config import add_argument, get_config

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description='Estimate the size and weight distribution of kiwifruits using Bayesian and simulation-based approaches.'
)
    
add_argument(parser, "--split", 0, "Whether to perform a train-test split on the full dataset, overriding the provided train-test data")

config = get_config("../..", parser)

##########################################################################################################
### Constants
##########################################################################################################

##########################################################################################################
### Pipeline
##########################################################################################################

def main():
    print("hello world.")

if __name__ == "__main__":
    main()
     