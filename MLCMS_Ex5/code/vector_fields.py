import pandas as pd
import numpy as np
import math
from typing import Tuple
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from function_approximation import rbf_derivative, nonlinear_least_squares

def fetch_vector_data(dir_path="../data/", base_filename="linear_vectorfield_data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetches and returns vector field data from specified files.
    
    :param dir_path: Path of the directory containing the files
    :param base_filename: Common part of the file names; suffixes "_x0.txt" and "_x1.txt" are added to fetch initial and final data
    :returns: Tuple containing initial and final data in the form of numpy ndarrays
    """
    initial_data = pd.read_csv(f'{dir_path}{base_filename}_x0.txt', sep=' ', header=None).to_numpy()
    final_data = pd.read_csv(f'{dir_path}{base_filename}_x1.txt', sep=' ', header=None).to_numpy()
    return initial_data, final_data

def compute_vector_field(time_delta: float, initial_data=None, final_data=None) -> np.ndarray:
    """
    Computes and returns vector field approximation using finite-difference formula.
    
    :param time_delta: Time difference used in finite-difference formula
    :param initial_data: Data at the beginning of the time interval; if None, data is fetched from file
    :param final_data: Data at the end of the time interval; if None, data is fetched from file
    :returns: Approximated vector field as a numpy ndarray
    """
    if initial_data is None or final_data is None:
        initial_data, final_data = fetch_vector_data()
    return (final_data - initial_data) / time_delta

def create_phase_portrait_matrix(A: np.ndarray, title_suffix: str, save_plots=False,
                                 save_path: str = None, display=True):
    """
    Plots the phase portrait of the linear system Ax
    :param A: system's (2x2 matrix in our case)
    :param title_suffix: suffix to add to the title (after the value of alpha and A's eigenvalues
    :param save_plots: if True, saves the plots instead of displaying them
    :param save_path: path where to save the plots if save_plots is True
    :param display: if True, display the plots
    """
    w = 10  # width
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    eigenvalues = np.linalg.eigvals(A)
    print("Eigenvalues of A: ", eigenvalues)
    # linear vector field A*x
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)
    fig = plt.figure(figsize=(5, 5))
    plt.streamplot(X, Y, U, V, density=1.0)
    if display:
        plt.show()
    if save_plots:
        plt.savefig(save_path)
def calculate_trajectory(initial_data, final_data, derivative_func, args, search_best_delta_t=False, final_time=0.1, generate_plot=False):
    """
    Solves the initial value problem for a whole dataset of points, up to a certain moment in time.
    
    :param initial_point: Data at the beginning
    :param final_point: Data at the end
    :param function: Function for getting derivative for next steps generation
    :param arguments: Additional arguments to be passed to the function
    :param find_best_time_difference: If True, also the time difference where we have lowest MSE is searched
    :param end_time: End time for the simulation
    :param plot: If True, produces a scatter plot of the trajectory (orange) with the final points in blue
    :returns: Points at time end_time, best point in time (getting lowest MSE), lowest MSE
    """
    best_delta_t = -1
    min_mse = math.inf
    predicted_final_data = []
    time_eval = np.linspace(0, final_time, 100)
    solution_list = []
    
    for index in range(len(initial_data)):
        solution = solve_ivp(derivative_func, [0, final_time], initial_data[index], args=args, t_eval=time_eval)
        predicted_final_data.append([solution.y[0, -1], solution.y[1, -1]])
        
        if search_best_delta_t:
            solution_list.append(solution.y)
            
        if generate_plot:
            plt.scatter(final_data[index, 0], final_data[index, 1], c='blue', s=10)
            plt.scatter(solution.y[0, :], solution.y[1, :], c='orange', s=4)
            
    if search_best_delta_t:
        for i in range(len(time_eval)):
            pred = [[solution_list[el][0][i], solution_list[el][1][i]] for el in range(len(solution_list))]
            mse = np.mean(np.linalg.norm(pred - final_data, axis=1)**2)
            if mse < min_mse:
                min_mse = mse
                best_delta_t = time_eval[i]
                
    if generate_plot:
        plt.rcParams["figure.figsize"] = (14,14)
        plt.show()
        
    return predicted_final_data, best_delta_t, min_mse


def find_optimal_rbf_config(initial_data, final_data, delta_t=0.01, final_time=0.05):
    """
    Conducts grid search over various different eps and n_bases values, returning the whole configuration with lowest MSE.
    
    :param initial_point: Data at the beginning
    :param final_point: Data at the end
    :param time_difference: Time difference to approximate the vector field between initial_point and final_point
    :param end_time: Total time of solve_ivp system solving trajectory
    :return: Best mse found with the configuration, including eps, n_bases, time_difference at which the mse was found, centers
    """
    optimal_mse, optimal_eps, optimal_n_bases, optimal_delta_t = math.inf, -1, -1, -1
    n_bases_trials = [int(i) for i in np.linspace(100, 1001, 20)]
    
    for n_bases in n_bases_trials:
        centers = initial_data[np.random.choice(range(initial_data.shape[0]), replace=False, size=n_bases)]
        
        for eps in (0.3, 0.5, 0.7, 1.0, 5.0, 15, 25):
            v = compute_vector_field(delta_t, initial_data, final_data)
            C, res, _, _, _, eps, phi = nonlinear_least_squares(data=(initial_data, v), n_bases=n_bases, eps=eps, centers=centers)
            pred_final_data, best_delta_t, best_mse = calculate_trajectory(initial_data, final_data, rbf_derivative, search_best_delta_t=True, args=[centers, eps, C], final_time=final_time)

            
            if optimal_mse > best_mse:
                optimal_mse, optimal_eps, optimal_n_bases, optimal_delta_t, optimal_centers  = best_mse, eps, n_bases, best_delta_t, centers
                
    print(f"Optimal configuration: eps = {optimal_eps}, n_bases = {optimal_n_bases}, dt = {optimal_delta_t}, giving MSE = {optimal_mse}")
    
    return optimal_mse, optimal_eps, optimal_n_bases, optimal_delta_t, optimal_centers


def create_phase_portrait_derivative(funct, args, title_suffix: str, save_plots=False,
                                     save_path: str = None, display=True, fig_size=5, w=4.5):
    """
    Plots the phase portrait given a 'funct' that gives the derivatives for a certain point
    :param funct: given a 2d point gives back the 2 derivatives
    :param title_suffix: suffix to add to the title (after the value of alpha and A's eigenvalues
    :param save_plots: if True, saves the plots instead of displaying them
    :param save_path: path where to save the plots if save_plots is True
    :param display: if True, display the plots
    :param fig_size: gives width and height of plotted figure
    :param w: useful for defining range for setting Y and X
    """
    # setting up grid width/height
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    # dynamic system parameter, responsible for the change in behaviour
    U, V = [], []
    for x2 in X[0]:
        for x1 in Y[:, 0]:
            res = funct(0, np.array([x1, x2]), *args)
            U.append(res[0][0])
            V.append(res[0][1])
    U = np.reshape(U, X.shape)
    V = np.reshape(V, X.shape)
    plt.figure(figsize=(fig_size, fig_size))
    plt.streamplot(X, Y, U, V, density=2)
    plt.title(f"{title_suffix}")
    if display:
        plt.show()
    if save_plots:
        plt.savefig(save_path)

