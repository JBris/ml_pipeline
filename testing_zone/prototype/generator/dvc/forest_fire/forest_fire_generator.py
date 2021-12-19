##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse
import configparser
import numpy as np
import os
import pandas as pd
import yaml

# Internal

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Generate a synthetic dataset.'
)

def _add_argument(parser: argparse.ArgumentParser, name: str, default: str, arg_help: str, type = int) -> None:
    parser.add_argument(
        name,
        default=default,
        help=f"{arg_help}. Defaults to '{default}'.",
        type=type
    )

_add_argument(parser, "--type", "day", "The type of synthetic dataset", str)
_add_argument(parser, "--n", 5, "The number of samples to take", int)
_add_argument(parser, "--mu", 5, "The mean of feature x2", float)

args = parser.parse_args()

##########################################################################################################
### Constants
##########################################################################################################

CONFIG = configparser.ConfigParser()
CONFIG.read('../../../config/forest_fires/forest_fires.ini')
PARAMS = yaml.safe_load(open("params.yaml"))["forest_fires"]

##########################################################################################################
### Main
##########################################################################################################

class ForestFiresGenerator:
    def generate(self, input, output, args: argparse.Namespace) -> None:
        df_sample = self.sample(input, args)
        wind = df_sample.wind
        x1_iqr = np.percentile(wind, [25, 75])
        x1 = np.random.uniform(x1_iqr[0], x1_iqr[1], args.n)
        x2 = np.random.normal(args.mu, np.std(wind), args.n)
        df_output = pd.DataFrame({
            "x1": x1,
            "x2": x2,
            "wind": wind
        })
        os.makedirs(PARAMS["out_dir"], exist_ok=True)
        df_output.to_csv(f"{PARAMS['out_dir']}/forestfires.csv", index = False)

    def get_dataset(self, file) -> pd.DataFrame:
        return pd.read_csv(file)

    def sample(self, file, args: argparse.Namespace) -> pd.DataFrame:
        df = self.get_dataset(file)
        return df.sample(n = args.n, replace = True, axis = 0)

def main() -> None:
    input = CONFIG.get("files", "input")
    output = CONFIG.get("files", "output")
    print(output)
    generator = ForestFiresGenerator()
    generator.generate(input, output, args)
    
if __name__ == "__main__":
    main()
 