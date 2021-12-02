import argparse
import numpy as np
import pandas as pd

from .utils import _add_argument

class DayGenerator:

    def generate(self, files: dict, args: argparse.Namespace) -> None:
        df_sample = self.sample(files["day"], args)
        cnt = df_sample.cnt
        x1_iqr = np.percentile(cnt, [25, 75])
        x1 = np.random.uniform(x1_iqr[0], x1_iqr[1], args.n)
        x2 = np.random.normal(np.average(cnt), np.std(cnt), args.n)
        df_output = pd.DataFrame({
            "x1": x1,
            "x2": x2,
            "cnt": cnt
        })
        df_output.to_csv("../out/day_file.csv", index = False)

    def sample(self, file, args: argparse.Namespace) -> pd.DataFrame:
        df = pd.read_csv(file)
        return df.sample(n = args.n, replace = True, axis = 0)

    def add_args(self, parser : argparse.ArgumentParser):
        _add_argument(parser, "--n", 5, "The number of samples to take", int)