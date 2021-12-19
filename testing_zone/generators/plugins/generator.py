##########################################################################################################
### Imports  
##########################################################################################################

# External
import argparse

from yapsy.PluginManager import PluginManager

# Internal
from plugins.core.igenerator import IGenerator
from plugins.core.utils import _add_argument

##########################################################################################################
### Parameters
##########################################################################################################

parser = argparse.ArgumentParser(
    description = 'Generate a synthetic dataset.'
)

_add_argument(parser, "--type", "day", "The type of synthetic dataset", str)

##########################################################################################################
### Main
##########################################################################################################

def main() -> None:
    manager = PluginManager()
    manager.setPluginPlaces(["plugins/core"])
    manager.setCategoriesFilter({
        "Generator" : IGenerator
    })
    manager.collectPlugins()
    
    generator_map = {}
    for pluginInfo in manager.getAllPlugins(): 
        manager.activatePluginByName(pluginInfo.name)
        plugin_object = pluginInfo.plugin_object
        generator_map[plugin_object.type] = plugin_object
        plugin_object.add_args(parser)

    args = parser.parse_args()
    selected_generator = generator_map.get(args.type)
    if selected_generator is None:
        print(f"Invalid generator type: {args.type}")
        exit(2)

    selected_generator.generate(args)

if __name__ == "__main__":
    main()
