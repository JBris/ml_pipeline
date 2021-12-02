from argparse import ArgumentParser, Namespace

class DatasetGenerator:
    def __init__(self, generator_map : dict, files : dict) -> None:
        self.generator_map = generator_map
        self.files = files

    def get(self, type : str):
        return self.generator_map.get(type)

    def generate(self, type : str, args : Namespace):
        generator = self.get(type)
        if generator is None: return generator
        return generator.generate(self.files, args)

    def add_args(self, parser : ArgumentParser):
        for key, generator in self.generator_map.items():
            generator.add_args(parser)
        