"""
Machine Learning Pipeline Tuning

A library of hyperparameter tuning methods for the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

# External


##########################################################################################################
### Library  
##########################################################################################################

def to_grid_search(grid_config: dict) -> dict:
    """
    Perform hyperparameter tuning using grid search.

    Parameters
    --------------
    grid_config: dict
        The custom grid configuration dictionary.

    Returns
    ---------
    grid: dict
        The grid search grid.
    """
    grid = {}

    for model, params in grid_config.items():
        grid[model] = {}
        for param, values in params.items():
            grid[model][param] = values["value"]

    return grid

def get_random_search_distributions() -> dict:
    from scipy.stats import norm, uniform

    search_map = {
        "normal": norm,
        "uniform": uniform
    }
    return search_map

def to_random_search(grid_config: dict) -> dict:
    """
    Perform hyperparameter tuning using random search.

    Parameters
    --------------
    grid_config: dict
        The custom grid configuration dictionary.

    Returns
    ---------
    grid: dict
        The random search grid.
    """
    grid = {}
    distributions = get_random_search_distributions()

    for model, params in grid_config.items():
        grid[model] = {}
        for param, values in params.items():
            if "distribution" in values and "kwargs" in values:
                distribution_key = values["distribution"]
                distribution = distributions.get(distribution_key)
                if distribution is None:
                    raise Exception(f"Error: Invalid distribution passed to random search - {distribution_key}")
                grid[model][param] = distribution(**values["kwargs"])
            else:
                grid[model][param] = values["value"]

    return grid

def get_ray_tune_search_distributions():
    from ray.tune import choice, uniform, quniform, loguniform
    search_map = {
        "choice": choice,
        "uniform": uniform,
        "quniform": quniform,
        "loguniform": loguniform
    }
    return search_map

def to_ray_tune_grid(grid_config: dict):
    """
    Perform hyperparameter tuning using Ray Tune.

    Parameters
    --------------
    grid_config: dict
        The custom grid configuration dictionary.

    Returns
    ---------
    grid: dict
        The Ray Tune grid.
    """
    grid = {}
    distributions = get_ray_tune_search_distributions()

    for model, params in grid_config.items():
        grid[model] = {}
        for param, values in params.items():
            if "distribution" in values and "kwargs" in values:
                distribution_key = values["distribution"]
                distribution = distributions.get(distribution_key)
                if distribution is None:
                    raise Exception(f"Error: Invalid distribution passed to Ray Tune grid - {distribution_key}")
                grid[model][param] = distribution(**values["kwargs"])
            else:
                grid[model][param] = values["value"]

    return grid

GRID_TRANSFORMERS = {
    "grid": to_grid_search,
    "random": to_random_search,
    "tune-sklearn": to_ray_tune_grid
}

class GridTransformer:
    def __init__(self, search_algorithm: str, search_library: str) -> None:
        self.search_algorithm = search_algorithm
        self.search_library = search_library
        self.grid_transformers = GRID_TRANSFORMERS

    def transform(self, grid_config: dict) -> dict:
        grid_transformers = self.grid_transformers
        if self.search_library in grid_transformers:
            grid_transformer_func = grid_transformers.get(self.search_library) 
        else:
            grid_transformer_func = grid_transformers.get(self.search_algorithm) 

        if grid_transformer_func is None:
            return {}
        else:
            return grid_transformer_func(grid_config)
        