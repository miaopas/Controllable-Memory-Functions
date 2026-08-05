import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.linalg import toeplitz

def get_rho_lstsq(x, y, end_point=None, ydim=0):
    """
    Computes regression weights (rho) via least squares or linear regression.

    Args:
        x (np.ndarray): Input array of shape (data_num, path_length, ...).
        y (np.ndarray): Target array of shape (data_num, path_length, ...).
        end_point (int or None): If None, evaluates y at all time steps using Toeplitz design matrix.
                                 If given as a time step index, fits using entire x and y at that specific time step.
        ydim (int): Index of the target dimension to use from y.

    Returns:
        np.ndarray: Regression weights (rho) of shape (path_length, dim).
    """
    data_num, path_length = x.shape[0], x.shape[1]
    feature_dims = x.shape[2:]
    dim = np.prod(feature_dims)

    if end_point is None:
        Y = np.ravel(y[..., ydim], order="F")[..., None]

        # Flatten feature dimensions into a single dim
        x_flat = x.reshape(data_num, path_length, dim)  # shape: (data_num, path_length, dim)

        # Triangular matrix repeated per feature
        tri = np.tile(np.tri(path_length), (data_num, 1))  # (data_num * path_length, path_length)

        # Construct Toeplitz for each feature and combine
        X_list = []
        for d in range(dim):
            toeplitz_matrix = toeplitz(x_flat[..., d])[:, :path_length]  # (data_num * path_length, path_length)
            x_d = (toeplitz_matrix * tri).reshape(data_num, path_length, path_length).reshape(-1, path_length, order="F")
            X_list.append(np.flip(x_d, axis=-1))

        X = np.concatenate(X_list, axis=1)  # shape: (data_num * path_length, path_length * dim)

    else:
        # Flatten feature dimensions, keep (data_num, path_length * dim)
        X = x.reshape(data_num, path_length * dim)
        Y = y[:, end_point, ydim]  # Select y at specific time step
        Y = Y.ravel()  # Ensure 1D array for LinearRegression

    # Fit linear model
    reg = LinearRegression(n_jobs=6).fit(X, Y)
    rho = reg.coef_

    if end_point is not None:
        rho = rho.reshape(path_length, *feature_dims)
    else:
        rho = rho.reshape(dim, path_length).T
        rho = rho.reshape(path_length, *feature_dims)

    return np.flip(rho, axis=0)
