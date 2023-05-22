import numpy as np
import ot


def calculate_cost_matrix(
        target: np.ndarray, source: np.ndarray, target_dx: float = 1.0, source_dx: float = 1.0,
                          ):
    """
    target: 2D numpy array of elevations for the target landscape
    source: 2D numpy array of elevations for the source landscape
    target_dx: grid x,y spacing for target landscape
    source_dx : grid x,y, spacing for source landscape.
    :returns: Cost matrix associated with two landscapes
    """
    # Locations for source landscape
    source_xx = np.arange(0, source.shape[0], source_dx)
    source_yy = np.arange(0, source.shape[1], source_dx)
    source_X, source_Y = np.meshgrid(source_xx, source_yy)
    source_XY = np.array([source_X.flatten(), source_Y.flatten()]).T

    # Locations for target landscape
    target_xx = np.arange(0, target.shape[0], target_dx)
    target_yy = np.arange(0, target.shape[1], target_dx)
    target_X, target_Y = np.meshgrid(target_xx, target_yy)
    target_XY = np.array([target_X.flatten(), target_Y.flatten()]).T

    # Calculate cost matrix
    cost_matrix = ot.dist(source_XY, target_XY)  # Default is squared Euclidean distance

    return cost_matrix


def calculate_OT_loss(target: np.ndarray, source: np.ndarray, cost_matrix, max_num_iterations):
    """
    :param target: numpy array of elevations in target landscape
    :param source: numpy array of elevations in source landscape
    :param cost_matrix: cost matrix between target and source landscape
    :param max_num_iterations: Maximum number of iterations to calculate the OT loss
    :return: Minimum cost required to transport source to target
    """
    source_normalised = (source / np.sum(source)).flatten()
    target_normalised = (target / np.sum(target)).flatten()

    OT_loss: object = ot.emd2(source_normalised, target_normalised, cost_matrix, numItermax=max_num_iterations)
    return OT_loss


# Load in landscapes from file
t = np.load('target_example_10Myr_seed1_100m.npy')
s = np.load('source_example_10Myr_seed10_100m.npy')

# Run an example
C = calculate_cost_matrix(t, s)
print("Calculating loss function")
W = calculate_OT_loss(t, s, C, max_num_iterations=1e7)
print("W = ", W)
