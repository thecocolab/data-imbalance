import numpy as np
from typing import Tuple, Optional


def gaussian_binary(
    mean_distance: float = 2,
    standard_deviation: float = 1,
    n_samples_per_class: int = 1000,
    n_features: int = 1,
    n_groups: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Generates random gaussian distributions with two different means and corresponding class labels.

    Args:
        mean_distance (float): distance between the mean of the two generated distributions
        standard_deviation (float): standard deviation of the gaussian distributions
        n_samples_per_class (int): number of samples per class (total will be n_samples_per_class * 2)
        n_features (int): number of random variables (columns)
        n_groups (int): number of groups to assign to the data (set to 0 to disable groups)

    Returns:
        x (ndarray): the generated data (n_samples x n_features)
        y (ndarray): the corresponding labels as an integer array of 0 and 1
        groups (ndarray, None): group labels assigned to the data (None if n_groups is set to 0)
    """
    x, y, groups = [], [], []
    for group in range(n_groups):
        n_samples_per_group = n_samples_per_class // n_groups
        # generate random data
        class0 = np.random.normal(
            0, standard_deviation, size=(n_samples_per_group, n_features)
        )
        class1 = np.random.normal(
            mean_distance, standard_deviation, size=(n_samples_per_group, n_features)
        )

        x.append(np.concatenate([class0, class1]))

        # generate class labels
        y.append(
            np.concatenate(
                [
                    np.zeros(n_samples_per_group, dtype=int),
                    np.ones(n_samples_per_group, dtype=int),
                ]
            )
        )
        groups.append([group] * len(x[-1]))

    return (
        np.concatenate(x) if len(x) > 1 else x[0],
        np.concatenate(y) if len(y) > 1 else y[0],
        None if n_groups == 1 else np.concatenate(groups),
    )
