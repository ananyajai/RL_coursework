import itertools
import torch
from typing import Dict, List, Tuple, Iterable


def generate_hparam_configs(base_config:Dict, hparam_ranges:Dict) -> Tuple[List[Dict], List[str]]:
    """
    Generate a list of hyperparameter configurations for hparam sweeping

    :param base_config (Dict): base configuration dictionary
    :param hparam_ranges (Dict): dictionary mapping hyperparameter names to lists of values to sweep over
    :return (Tuple[List[Dict], List[str]]): list of hyperparameter configurations and swept parameter names
    """

    keys, values = zip(*hparam_ranges.items())
    hparam_configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    swept_params = list(hparam_ranges.keys())

    new_configs = []
    for hparam_config in hparam_configurations:
        new_config = base_config.copy()
        new_config.update(hparam_config)
        new_configs.append(new_config)

    return new_configs, swept_params


def grid_search(num_samples: int, min: float = None, max: float = None, **kwargs)->Iterable:
    """ Implement this method to set hparam range over a grid of hyperparameters.
    :param num_samples (int): number of samples making up the grid
    :param min (float): minimum value for the allowed range to sweep over
    :param max (float): maximum value for the allowed range to sweep over
    :param kwargs: additional keyword arguments to parametrise the grid.
    :return (Iterable): tensor/array/list/etc... of values to sweep over

    Example use: hparam_ranges['batch_size'] = grid_search(64, 512, 6, log=True)

    **YOU MAY IMPLEMENT THIS FUNCTION FOR Q5**

    """
    # raise NotImplementedError
    values = torch.zeros(num_samples)
    return values


def random_search(num_samples: int, distribution: str, min: float=None, max: float=None, **kwargs) -> Iterable:
    """ Implement this method to sweep via random search, sampling from a given distribution.
    :param num_samples (int): number of samples to take from the distribution
    :param distribution (str): name of the distribution to sample from
        (you can instantiate the distribution using torch.distributions, numpy.random, or else).
    :param min (float): minimum value for the allowed range to sweep over (for continuous distributions)
    :param max (float): maximum value for the allowed range to sweep over (for continuous distributions)
    :param kwargs: additional keyword arguments to parametrise the distribution.

    Example use: hparam_ranges['lr'] = random_search(1e-6, 1e-1, 10, distribution='exponential', lambda=0.1)

    **YOU MAY IMPLEMENT THIS FUNCTION FOR Q5**

    """
    # raise NotImplementedError
    values = torch.zeros(num_samples)

    values = []

    for _ in range(num_samples):
        if distribution == 'uniform':
            sample = torch.rand(1).item() * (max_val - min_val) + min_val
        elif distribution == 'normal':
            sample = torch.normal(mean=kwargs.get('mean', 0.0), std=kwargs.get('std', 1.0)).item()
        elif distribution == 'exponential':
            sample = torch.distributions.exponential.Exponential(kwargs.get('lambda', 1.0)).sample().item()
        elif distribution == 'gamma':
            sample = torch.distributions.gamma.Gamma(kwargs.get('alpha', 1.0), kwargs.get('beta', 1.0)).sample().item()
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        values.append(sample)

    return values

# base_config = {
#     "policy_learning_rate": 1e-4,
#     "critic_learning_rate": 1e-3,
#     "critic_hidden_size": [32, 32, 32],
#     "policy_hidden_size": [32, 32, 32],
#     "gamma": 0.99,
#     "tau": 0.5,
#     "batch_size": 32,
#     "buffer_capacity": int(1e6),
# }

# hparam_ranges = {
#     "policy_learning_rate": [1e-4, 1e-3, 1e-2],
#     "critic_learning_rate": [1e-4, 1e-3, 1e-2],
#     "critic_hidden_size": [[32, 32, 32], [64, 64, 64], [128, 128]],
#     "policy_hidden_size": [[32, 32, 32], [64, 64, 64], [128, 128]],
#     "gamma": [0.95, 0.99, 0.999],
#     "tau": [0.1, 0.5, 0.9],
#     "batch_size": [32, 64, 128],
#     "buffer_capacity": [int(1e5), int(1e6), int(1e7)],
# }

# configs, swept_params = generate_hparam_configs(base_config, hparam_ranges)
# print("Generated configurations:", configs)
# print("Swept parameters:", swept_params)
