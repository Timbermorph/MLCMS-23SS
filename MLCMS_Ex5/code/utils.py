import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Union, Iterable, Tuple
from scipy.spatial.distance import cdist


###Task1###
###Task2###
def load_datasets(file_path_1, file_path_2):
    """
    Load datasets from two files.

    Parameters:
    - file_path_1: str, file path of the first dataset
    - file_path_2: str, file path of the second dataset

    Returns:
    - x0: numpy.ndarray, first dataset
    - x1: numpy.ndarray, second dataset
    """
    x0 = np.loadtxt(file_path_1)
    x1 = np.loadtxt(file_path_2)
    return x0, x1


def estimate_vectors(x0, x1, dt):
    """
    Estimate the vectors v(k) and approximate the matrix A.

    Parameters:
    - x0: numpy.ndarray, initial dataset
    - x1: numpy.ndarray, next dataset
    - dt: float, time step between x0 and x1

    Returns:
    - A_hat: numpy.ndarray, estimated matrix A
    """
    v = (x1 - x0) / dt
    A_hat = np.linalg.lstsq(x0, v, rcond=None)[0]
    return A_hat


def plot_vector_field(A_hat, x_range, y_range, x0):
    """
    Plot the vector field defined by the matrix A_hat.

    Parameters:
    - A_hat: numpy.ndarray, matrix A
    - x_range: numpy.ndarray, range of x values
    - y_range: numpy.ndarray, range of y values
    - x0: numpy.ndarray, initial dataset
    """
    X, Y = np.meshgrid(x_range, y_range)
    U = A_hat[0, 0] * X + A_hat[0, 1] * Y
    V = A_hat[1, 0] * X + A_hat[1, 1] * Y

    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U, V, scale=20, color='b')
    plt.scatter(x0[:, 0], x0[:, 1], color='r', label='$x_0$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Vector Field')
    plt.legend()
    plt.show()


def calculate_mse(A_hat, x0, x1, Tend):
    """
    Calculate the mean squared error (MSE) of the estimated system.

    Parameters:
    - A_hat: numpy.ndarray, estimated matrix A
    - x0: numpy.ndarray, initial dataset
    - x1: numpy.ndarray, next dataset
    - Tend: float, end time for integration

    Returns:
    - mse: float, mean squared error
    """
    estimated_points = []
    for initial_point in x0:
        def linear_system(t, x):
            return A_hat.dot(x)

        sol = solve_ivp(linear_system, [0, Tend], initial_point, t_eval=[Tend])
        estimated_points.append(sol.y[:, -1])

    mse = (np.linalg.norm(estimated_points - x1) ** 2) / len(x0)
    return mse


def phase_portrait(A, X, Y, title=""):
    """
    Plot the phase portrait of a linear system.

    Parameters:
    - A: numpy.ndarray, matrix A
    - X: numpy.ndarray, x-coordinate grid values
    - Y: numpy.ndarray, y-coordinate grid values
    - title: str, title of the plot (optional)
    """
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)

    plt.streamplot(X, Y, U, V, color='b', linewidth=1, density=1.5, arrowstyle='->', arrowsize=1.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))
    plt.grid(True)


###Task3###
###Task4###
###Task5###
def rbf(x, x_l, eps):
    """
    radial basic function

    Parameters:
    - x: points
    - x_l: center/s
    - eps: radius of gaussian

    Returns:
    - matrix contains radial basic function
    """
    return np.exp(-cdist(x, x_l) ** 2 / eps ** 2)


def get_points_and_targets(data: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Depending on the type of the parameter 'data', returns correctly the points and the targets

    Parameters:
    - data: Iterable containing 2 numpy ndarrays: points and targets

    Returns:
    - points and targets
    """
    if len(data) != 2:
        raise ValueError(f"Parameter data must be an Iterable of 2 numpy ndarrays, got {len(data)} elements")

    points, targets = data[0], data[1]

    return points, targets


def compute_bf(points: np.ndarray, eps: float, n_bases: int, centers: np.ndarray = None):
    """
    Compute the basis functions

    Parameter:
    - points: the points on which to calculate the basis functions
    - centers: the center points to pick to compute the basis functions
    - eps: epsilon param of the basis functions
    - n_bases: number of basis functions to compute

    Returns:
    - list of basis functions evaluated on every point in 'points'
    """
    if centers is None:
        centers = points[np.random.choice(range(points.shape[0]), replace=False, size=n_bases)]
    phi = rbf(points, centers, eps)
    return phi, centers


def approx_nonlin_func(data: Iterable[np.ndarray] = "../data/nonlinear_function_data.txt", n_bases: int = 5, eps: float = 0.1,
                       centers: np.ndarray = None):
    """
    Approximate a non-linear function through least squares

    Returns:
    - tuple (least squares solution (transposed), residuals, rank of coefficients matrix,
             singular values of coefficient matrix, centers, eps and phi (list_of_basis))
    """
    # get coefficients and targets form the data
    points, targets = get_points_and_targets(data)

    # evaluate the basis functions on the whole data and putting each result in an array
    list_of_bases, centers = compute_bf(points=points, centers=centers, eps=eps, n_bases=n_bases)

    # solve least square using the basis functions with nonlinear function
    sol, residuals, rank, singvals = np.linalg.lstsq(a=list_of_bases, b=targets, rcond=1e-5)
    return sol, residuals, rank, singvals, centers, eps, list_of_bases


def rbf_approx(t, y, centers, eps, C):
    """
    function to return vector field of a single point (rbf)

    Parameters:
    - t: time (for solve_ivp)
    - y: single point
    - centers: all centers
    - eps: radius of gaussians
    - C: coefficient matrix, found with least squares

    Return:
    - derivative for point y
    """
    y = y.reshape(1, y.shape[-1])
    phi = np.exp(-cdist(y, centers) ** 2 / eps ** 2)
    return phi @ C