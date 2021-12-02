import argparse
import numpy as np
import pandas as pd

from plugins.core.igenerator import IGenerator
from plugins.core.utils import _add_argument

class ForestFiresGenerator(IGenerator):
    def __init__(self):
        super(ForestFiresGenerator, self).__init__()
        self.type = "forest_fires"
        self.file = f"{self.dir}/forestfires.csv"

    def generate(self, args: argparse.Namespace) -> None:
        n = 10
        df_sample = self.sample(args, n)
        wind = df_sample.wind
        x1_iqr = np.percentile(wind, [25, 75])
        x1 = np.random.uniform(x1_iqr[0], x1_iqr[1], n)
        x2 = np.random.normal(args.mu, np.std(wind), n)
        df_output = pd.DataFrame({
            "x1": x1,
            "x2": x2,
            "wind": wind
        })
        df_output.to_csv("../out/forestfires.csv", index = False)

    def sample(self, args: argparse.Namespace, n: int = 10) -> pd.DataFrame:
        df = pd.read_csv(self.file)
        return df.sample(n = n, replace = True, axis = 0)

    def add_args(self, parser : argparse.ArgumentParser):
        _add_argument(parser, "--mu", 5, "The mean of feature x2", float)
