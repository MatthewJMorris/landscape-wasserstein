import numpy as np
import ot


def calculate_objective_function(
    target: np.ndarray, source: np.ndarray, mu: float, dx: float = 1.0, dy: float = 1.0
) -> float:
    """
    A function which calculates H, the objective function, as defined in Equation 3.
    H(t,s) = w2x(t,s)**2 + w2y(t,s)**2 + mu*P**2, where P = difference in mean elevations of t and s.
    target = 2D numpy array of elevations of target landscape
    source = 2D numpy array of elevations of source landscape
    mu = Scaling factor, float
    dx, dy =  x-step and y-step of landscape arrays (default = 1)

    Returns: the objective function value
    """

    # Calculate marginals
    targ_x = np.sum(target, axis=0)
    targ_x_norm = targ_x / np.sum(targ_x)  # Weights of first distribution
    targ_y = np.sum(target, axis=1)
    targ_y_norm = targ_y / np.sum(targ_y)  # Weights of second distribution

    srce_x = np.sum(source, axis=0)
    srce_x_norm = srce_x / np.sum(srce_x)
    srce_y = np.sum(source, axis=1)
    srce_y_norm = srce_y / np.sum(srce_y)

    xx = np.arange(np.size(targ_x)) * dx  # Locations of x nodes
    yy = np.arange(np.size(targ_y)) * dy  # Locations of y nodes

    # Calculate wasserstein distances for the x and y summed distributions seperately. 
    # Returns squared Wasserstein distance
    w2x2 = ot.wasserstein_1d(xx, xx, targ_x_norm, srce_x_norm, p=2)
    w2y2 = ot.wasserstein_1d(yy, yy, targ_y_norm, srce_y_norm, p=2)

    z_target = np.mean(target.flatten())
    z_src = np.mean(source.flatten())
    P = z_target - z_src

    H = w2x2 + w2y2 + mu * (P**2)

    return H
    
    
# Load in landscapes from file
t = np.load('target_example_10Myr_seed1_100m.npy')
s = np.load('source_example_10Myr_seed10_100m.npy')

# Run an example
H_st = calculate_objective_function(t,s,mu=10000)
print("H = ",H_st)
