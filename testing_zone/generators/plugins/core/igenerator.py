from argparse import Namespace
import argparse
from yapsy.IPlugin import IPlugin

class IGenerator(IPlugin):
    def __init__(self):
        super(IGenerator, self).__init__()
        self.type = None
        self.dir = "../data"
        self.file = f"{self.dir}"
    
    def generate(self, args: Namespace):
        raise NotImplementedError("Generator plugin must implement a generator method.")

    def add_args(self, parser : argparse.ArgumentParser):
        return parser
    