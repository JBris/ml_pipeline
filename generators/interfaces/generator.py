##########################################################################################################
### Imports  
##########################################################################################################

# External
import abc
import argparse
import configparser
import numpy as np
import pandas as pd

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
CONFIG.read('config/days.ini')

##########################################################################################################
### Main
##########################################################################################################

class IGenerator(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def generate(self, input, output, args: argparse.Namespace) -> None:
        pass

class DayGenerator(IGenerator):
    def generate(self, input, output, args: argparse.Namespace) -> None:
        df_sample = self.sample(input, args)
        cnt = df_sample.cnt
        x1_iqr = np.percentile(cnt, [25, 75])
        x1 = np.random.uniform(x1_iqr[0], x1_iqr[1], args.n)
        x2 = np.random.normal(np.average(cnt), np.std(cnt), args.n)
        df_output = pd.DataFrame({
            "x1": x1,
            "x2": x2,
            "cnt": cnt
        })
        df_output.to_csv(output, index = False)

    def sample(self, file, args: argparse.Namespace) -> pd.DataFrame:
        df = pd.read_csv(file)
        return df.sample(n = args.n, replace = True, axis = 0)

def main() -> None:
    input = CONFIG.get("files", "input")
    output = CONFIG.get("files", "output")
    print(output)
    generator = DayGenerator()
    generator.generate(input, output, args)
    
if __name__ == "__main__":
    main()
