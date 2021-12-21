##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import os, sys

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

# Internal 
from pipeline_lib.config import add_argument, get_config

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Fit an ensemble estimator using PyCaret'
)
    
add_argument(parser, "--example", 0, "An example argument")

##########################################################################################################
### Constants
##########################################################################################################

CONFIG = get_config(base_dir, parser)

##########################################################################################################
### Pipeline
##########################################################################################################

def main():
    example = CONFIG.get('example')
    with open("data/example.txt", "w") as f: 
        f.write(f"Example: {example}")
    print(f"Created example.txt" )

if __name__ == "__main__":
    main()
     