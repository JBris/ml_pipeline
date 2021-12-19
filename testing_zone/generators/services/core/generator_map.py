from .forest_fire_generator import ForestFiresGenerator
from .day_generator import DayGenerator

generator_map = {
    "day": DayGenerator(),
    "forest_fire": ForestFiresGenerator()
}
