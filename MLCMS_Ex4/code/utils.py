import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from typing import Tuple, List

def solve_euler(f_ode, y0, time):
    """
    Solves the given ODE system in f_ode using forward Euler.
    :param f_ode: the right hand side of the ordinary differential equation d/dt x = f_ode(x(t)).
    :param y0: the initial condition to start the solution at.
    :param time: np.array of time values (equally spaced), where the solution must be obtained.
    :returns: (solution[time,values], time) tuple.
    """
    yt = np.zeros((len(time), len(y0)))
    yt[0, :] = y0
    step_size = time[1]-time[0]
    for k in range(1, len(time)):
        yt[k, :] = yt[k-1, :] + step_size * f_ode(yt[k-1, :])
    return yt, time


def plot_phase_portrait(A, X, Y):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    """
    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title('Streamplot for linear vector field A*x');
    ax0.set_aspect(1)
    return ax0


###Task1###
#draw phase portraits
def draw_phase_portrait(matrix: np.ndarray, alpha: float, additional_title: str, path: str = None):
    """
    Function to draw the phase portrait of a 2D linear system.
    
    @param matrix: The 2D matrix that describes the system.
    @param alpha: A parameter value, to be displayed in the plot title.
    @param additional_title: An additional string to add to the plot title.
    @param path: Optional parameter to specify a path to save the plot. If None, the plot is shown but not saved.
    """

    # Define the boundaries of the grid where the phase portrait is to be drawn
    grid_bound = 5
    # Create a grid of points within these boundaries
    Y_grid, X_grid = np.mgrid[-grid_bound:grid_bound:100j, -grid_bound:grid_bound:100j]

    # Compute the eigenvalues of the system's matrix. These are used in the plot title.
    eigvals = np.linalg.eigvals(matrix)
    print(f"Eigenvalues of the matrix: {eigvals}")

    # Compute the linear vector field at each point in the grid by multiplying the system's matrix with the point's coordinates
    UV_field = matrix @ np.vstack([X_grid.ravel(), Y_grid.ravel()])
    # Reshape the vector field arrays to match the grid's shape
    U_field = UV_field[0, :].reshape(X_grid.shape)
    V_field = UV_field[1, :].reshape(Y_grid.shape)

    # Create a new plot
    fig, ax = plt.subplots(figsize=(5, 5))
    # Draw a streamplot, i.e., plot the vector field as streamlines
    ax.streamplot(X_grid, Y_grid, U_field, V_field, density=.7)
    # Set the plot title to include alpha and the eigenvalues
    ax.set_title(f"alpha: {alpha}, Lambda 1: {eigvals[0]:.4f}, Lambda 2: {eigvals[1]:.4f} - {additional_title}")
    
    # If a path is provided, save the plot to this path. Otherwise, display the plot.
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
        
###Task3###

# Define the system

def dynamic_system(states: Tuple[float, float], t: float, alpha: float) -> List[float]:
    """
    Defines the dynamic system, or the behavior of the system over time.

    @param states: Tuple containing the current state variables (x1, x2).
    @param t: Time parameter (not used in this function but required for odeint solver compatibility).
    @param alpha: Control parameter of the system.

    @return: List of new states [new_x1, new_x2] calculated using the given dynamic system equations.
    """
    x1, x2 = states
    return [alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2), x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)]


#plot the system function
def compute_and_plot(system_func, alphas, title_suffixes, w=3, step=0.1):
    """
    Computes and plots a stream plot of the provided system function 
    for different values of alpha (system parameters).

    @param system_func: A function that represents the system dynamics.
    @param alphas: A list of system parameter values for which the stream plots are to be drawn.
    @param title_suffixes: A list of suffixes to be added to the plot title for each alpha value.
    @param w: Width of the square grid. Default is 3.
    @param step: Step size for the grid points. Default is 0.1.
    """

    # Create a grid of points. The grid is a square with side length 2w,
    # and the points are spaced 'step' distance apart.
    x = np.arange(-w, w, step)
    y = np.arange(-w, w, step)
    X, Y = np.meshgrid(x, y)

    # Iterate over all the alpha values
    for i, alpha in enumerate(alphas):
        # Create empty arrays for the U and V components of each vector
        U, V = np.zeros(X.shape), np.zeros(Y.shape)

        # Iterate over all the points on the grid
        for idx, _ in np.ndenumerate(X):
            # Apply the system function to each point on the grid
            u, v = system_func([X[idx], Y[idx]], 0, alpha)

            # Store the resulting vector in the U and V arrays
            U[idx] = u
            V[idx] = v

        # Create a new figure
        plt.figure()

        # Create a stream plot using the U and V arrays
        plt.streamplot(X, Y, U, V, density=2)

        # Set the title of the plot, using the current alpha value and its corresponding title suffix
        plt.title(f"alpha: {alpha} - {title_suffixes[i]}")

        # Display the plot
        plt.show()
        
#plot cusp bifurcation
def plot_cusp_bifurcation(alpha_two_limit: float, x_limit: float, n: int):
    """
    Function to plot the cusp bifurcation in 2D and 3D.
    @param alpha_two_limit: value to delimit the upper bound of alpha2 sampling (0, alpha_two_limit)
    @param x_limit: value to delimit the upper and lower bound of x sampling (-x_limit, x_limit)
    @param n: number of samples to get
    """
    # Sample a2 and x uniformly
    a2_samples = np.random.uniform(0, alpha_two_limit, n)
    x_samples = np.random.uniform(-x_limit, x_limit, n)
    a1_samples = []

    # Prepare dictionary for storing solutions
    solutions = {}

    # Calculate a1 and store solutions
    for x, a2 in zip(x_samples, a2_samples):
        a1 = -a2 * x + x ** 3
        key = (np.around(a1, decimals=3), np.around(a2, decimals=3))
        if key in solutions:
            solutions[key].add(np.around(x, decimals=3))
        else:
            solutions[key] = {np.around(x, decimals=3)}
        a1_samples.append(a1)

    # Determine colors based on the number of solutions
    colors = ["red" if len(solutions[(np.around(a1, decimals=3), np.around(a2, decimals=3))]) > 1 else "blue" for a1, a2 in zip(a1_samples, a2_samples)]

    # Create 3D plot
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a1_samples, a2_samples, x_samples, cmap='viridis', c=a2_samples)
    ax.set_xlabel("alpha_1")
    ax.set_ylabel("alpha_2")
    ax.set_zlabel("x")
    plt.show()

    # Create 2D plot
    plt.scatter(a1_samples, a2_samples, s=0.5, c=colors)
    plt.xlabel("alpha_1")
    plt.ylabel("alpha_2")
    plt.show()


###Task4###
###part1###
def logistic_map(r, x):
    '''
    Define logistic map
    '''
    return r * x * (1 - x)

def bifurcation_diagram(r_values, iterations, discard):
    '''
    bifurcation_diagram generates a bifurcation diagram for the logistic map. 
    r_values: An array or list of parameter values for the logistic map. These values determine the range of the bifurcation diagram.
    iterations: The total number of iterations to perform for each parameter value.
    discard: The number of initial iterations to discard to allow the system to reach a steady-state or a periodic orbit.
    '''
    steady_states = []
    x_values = []

    for r in r_values:
        x = 0.5  # Initial condition
        for _ in range(discard):
            x = logistic_map(r, x)
        
        x_values.append([])
        for _ in range(iterations):
            x = logistic_map(r, x)
            x_values[-1].append(x)
            if _ >= iterations - discard:
                steady_states.append(x)

    return steady_states, x_values

def plot_cobweb(logistic, r, x0, n):
    '''
    plot_cobweb function is used to create a cobweb plot for the logistic map. A cobweb plot visualizes the iterated function on a graph, highlighting the behavior of the system over multiple iterations.
    logistic: The logistic map function.
    r: The parameter value for the logistic map.
    x0: The initial condition or starting value.
    n: The number of iterations.
    '''
    # Plot the function and the y=x diagonal line.
    t = np.linspace(0, 1)
    plt.plot(t, logistic(r, t), 'k', lw=2)
    plt.plot([0, 1], [0, 1], 'k', lw=2)

    # Recursively apply y=f(x) and plot two lines:
    # (x, x) -> (x, y)
    # (x, y) -> (y, y)
    x = x0
    for i in range(n):
        y = logistic(r, x)
        # Plot the two lines.
        plt.plot([x, x], [x, y], 'k', lw=1)
        plt.plot([x, y], [y, y], 'k', lw=1)
        # Plot the positions with increasing opacity.
        plt.plot([x], [y], 'ok', ms=10, alpha=(i + 1) / n)
        x = y

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"$r={r:.1f}, \, x_0={x0:.1f}$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

###Part2###
# Define the Lorenz system
def lorenz_system(t, xyz, sigma, beta, rho):
    '''
    The lorenz_system function represents the mathematical model of the Lorenz system, which describes the behavior of a simplified atmospheric convection model. 
    t: The current time.
    xyz: The system state, represented by a list or array containing the values of x, y, and z.
    sigma: The parameter representing the Prandtl number.
    beta: The parameter representing a geometric factor.
    rho: The parameter representing the Rayleigh number.
    '''
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]
