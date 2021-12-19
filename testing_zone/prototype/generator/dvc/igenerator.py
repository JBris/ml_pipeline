import abc
import argparse
import pandas as pd

class IGenerator(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def get_dataset(self, file) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def generate(self, input, output, args: argparse.Namespace) -> None:
        pass