import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown, Button, HBox, VBox, Layout,Dropdown, Button, Layout, VBox, HBox
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.optimize import curve_fit
from scipy.stats import t

import plotly.express as px
import plotly.graph_objects as go

import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
from scipy.spatial import KDTree



### 1.Load Dataset###

def Load_data(file_paths):
    """
    Loads data from CSV files into Pandas DataFrames.

    Args:
        file_paths (list): List of file paths for CSV files.

    Returns:
        list: List of Pandas DataFrames containing the loaded data.
    """
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=' ', header=None)
        df.columns = ['id', 'frame', 'x', 'y', 'z']
        # creates a new column 'coords' in the DataFrame df, which contains the tuples of coordinates (x, y, z) for each pedestrian.
        df['coords'] = list(zip(df['x'], df['y'], df['z']))
        dataframes.append(df)
    return dataframes
    
def visualize_trajectories_interactive(data):
    """
    Provides an interactive visualization for trajectories.

    Args:
        data (DataFrame): DataFrame containing trajectory data.

    Returns:
        None
    """
    # Create a list of all pedestrians and add 'All' option
    all_pedestrians = sorted(data['id'].unique())
    all_pedestrians.insert(0, 'All')

    selected_pedestrians = []

    # Define the callback functions for the buttons
    def show_all_trajectories(button):
        visualize_trajectories(data, selected_pedestrians=None)

    def select_pedestrians(button):
        selected_pedestrian_widget = Dropdown(options=all_pedestrians, value='All', description='Select Pedestrian:', layout=Layout(width='500px'))
        add_button = Button(description='Add Pedestrian')
        selected_pedestrians_widget = []

        def add_pedestrian(add_button):
            selected_pedestrian = selected_pedestrian_widget.value
            if selected_pedestrian not in selected_pedestrians_widget:
                selected_pedestrians_widget.append(selected_pedestrian)
                visualize_trajectories(data, selected_pedestrians=selected_pedestrians_widget)

        add_button.on_click(add_pedestrian)

        display(VBox([selected_pedestrian_widget, add_button]))

    # Create buttons for showing all trajectories and selecting pedestrians
    show_all_button = Button(description='Show All Trajectories')
    select_button = Button(description='Select Pedestrians')

    # Connect the buttons with their callback functions
    show_all_button.on_click(show_all_trajectories)
    select_button.on_click(select_pedestrians)

    # Display the buttons
    display(HBox([show_all_button, select_button]))

def visualize_trajectories(data, selected_pedestrians=None):
    """
    Visualizes trajectories.

    Args:
        data (DataFrame): DataFrame containing trajectory data.
        selected_pedestrians (list): List of selected pedestrians' IDs.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if selected_pedestrians is None or 'All' in selected_pedestrians:
        # Plot all trajectories without labels and assign different colors to each trajectory
        unique_pedestrians = data['id'].unique()
        num_pedestrians = len(unique_pedestrians)
        colors = plt.cm.get_cmap('tab10', num_pedestrians)
        for i, pedestrian in enumerate(unique_pedestrians):
            pedestrian_data = data[data['id'] == pedestrian]
            ax.plot(pedestrian_data['x'], pedestrian_data['y'], pedestrian_data['z'], color=colors(i))
    else:
        # Plot selected pedestrians' trajectories with labels and assign a single color to all trajectories
        pedestrian_colors = 'red'
        for pedestrian in selected_pedestrians:
            pedestrian_data = data[data['id'] == pedestrian]
            ax.plot(pedestrian_data['x'], pedestrian_data['y'], pedestrian_data['z'], color=pedestrian_colors, label=pedestrian)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if selected_pedestrians is not None and 'All' not in selected_pedestrians:
        # Add a legend if not showing all trajectories
        ax.legend()

    # Show the plot
    plt.show()

def plot_pedestrian_positions(dataframes, frame):
    x = []
    y = []
    for df in dataframes:
        frame_data = df[df['frame'] == frame]
        for coord in frame_data['coords']:
            x.append(coord[0])
            y.append(coord[1])
    plt.scatter(x, y)
    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')
    plt.title('Pedestrian Positions at Frame {}'.format(frame))
    plt.show()
    
###2. Data Processing###

def load_data(file_path):
    """
    Loads data from a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): File path for the CSV file.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the loaded data.
    """
    df = pd.read_csv(file_path, sep=' ', header=None)
    df.columns = ['id', 'frame', 'x', 'y', 'z']
    df['coords'] = list(zip(df['x'], df['y'], df['z']))
    return df

def calculate_speed(df, frame_rate):
    """
    Calculates the speed for each pedestrian based on the change in position.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing the pedestrian data.
        frame_rate (float): Frame rate of the dataset.

    Returns:
        pd.DataFrame: Pandas DataFrame with speed for each pedestrian.
    """
    df['time'] = df['frame'] * frame_rate
    df['time_diff'] = df['time'].diff()
    df['dist_diff'] = np.sqrt((df['x'].diff())**2 + (df['y'].diff())**2)
    df['speed'] = df['dist_diff'] / df['time_diff']
    df['speed'] = df['speed'].abs()  # Take the absolute value of speed to ensure positive values
    df.drop(['time', 'time_diff', 'dist_diff'], axis=1, inplace=True)
    return df


import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.spatial import KDTree

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import t

def load_data(file_path):
    """
    Loads data from a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): File path for the CSV file.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the loaded data.
    """
    df = pd.read_csv(file_path, sep=' ', header=None)
    df.columns = ['id', 'frame', 'x', 'y', 'z']
    return df

import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import t

import numpy as np

# def calculate_mean_spacing(df, frame_rate, measure_interval):
#     frame_diff = int(frame_rate * measure_interval)
#     k = 10
#     unique_frames = df['frame'].unique()
#     mean_spacing_list = []
#     x_neighbor_columns = [f'x{i}' for i in range(1, k + 1)]
#     y_neighbor_columns = [f'y{i}' for i in range(1, k + 1)]

#     for frame in unique_frames:
#         if frame_diff == 0 or frame % frame_diff == 0:
#             frame_df = df[df['frame'] == frame]

#             if len(frame_df) < k + 1:
#                 continue

#             positions = frame_df[['x', 'y']].values
#             kdtree = KDTree(positions)

#             mean_spacing_frame = []
#             x_neighbors_frame = [[] for _ in range(k)]
#             y_neighbors_frame = [[] for _ in range(k)]

#             for i, position in enumerate(positions):
#                 _, indices = kdtree.query(position, k=k + 1)  # Finding k nearest neighbors and the point itself

#                 if len(indices) < k + 1:
#                     continue

#                 if np.max(indices[1:]) >= len(positions):
#                     continue

#                 neighbors = positions[indices[1:k+1]]  # Select the 10 nearest neighbors

#                 mean_distance = np.mean(np.linalg.norm(neighbors - position, axis=1))

#                 mean_spacing_frame.append(mean_distance)

#                 for j in range(k):
#                     x_neighbors_frame[j].append(neighbors[j, 0])
#                     y_neighbors_frame[j].append(neighbors[j, 1])

#             mean_spacing_list.extend(mean_spacing_frame)
#             for j in range(k):
#                 if len(x_neighbors_frame[j]) < len(df):
#                     avg_neighbor_x = np.mean(x_neighbors_frame[j])
#                     x_neighbors_frame[j].extend([avg_neighbor_x] * (len(df) - len(x_neighbors_frame[j])))

#                 if len(y_neighbors_frame[j]) < len(df):
#                     avg_neighbor_y = np.mean(y_neighbors_frame[j])
#                     y_neighbors_frame[j].extend([avg_neighbor_y] * (len(df) - len(y_neighbors_frame[j])))

#                 df[x_neighbor_columns[j]] = x_neighbors_frame[j]
#                 df[y_neighbor_columns[j]] = y_neighbors_frame[j]



#     if len(mean_spacing_list) < len(df):
#         mean_spacing_list.extend([np.nan] * (len(df) - len(mean_spacing_list)))

#     df['mean_spacing'] = mean_spacing_list

#     return df


def calculate_mean_spacing(df, frame_rate, measure_interval):
    """
    Calculate the mean spacing for each point in the DataFrame using the k nearest neighbors.
    
    Parameters:
        df (DataFrame): Input DataFrame containing position data.
        frame_rate (float): Frame rate of the data.
        measure_interval (float): Time interval in seconds for measuring mean spacing.
        
    Returns:
        DataFrame: Updated DataFrame with mean spacing values.
    """
    frame_diff = int(frame_rate * measure_interval)
    k = 10
    unique_frames = df['frame'].unique()
    mean_spacing_list = []
    x_neighbor_columns = [f'x{i}' for i in range(1, k + 1)]
    y_neighbor_columns = [f'y{i}' for i in range(1, k + 1)]

    for frame in unique_frames:
        if frame_diff == 0 or frame % frame_diff == 0:
            frame_df = df[df['frame'] == frame]

            if len(frame_df) < k + 1:
                continue

            positions = frame_df[['x', 'y']].values
            kdtree = KDTree(positions)

            mean_spacing_frame = []
            x_neighbors_frame = [[] for _ in range(k)]
            y_neighbors_frame = [[] for _ in range(k)]

            for i, position in enumerate(positions):
                _, indices = kdtree.query(position, k=k + 1)  # Finding k nearest neighbors and the point itself

                if len(indices) < k + 1:
                    continue

                if np.max(indices[1:]) >= len(positions):
                    continue

                neighbors = positions[indices[1:k+1]]  # Select the 10 nearest neighbors

                mean_distance = np.mean(np.linalg.norm(neighbors - position, axis=1))
                std_distance = np.std(np.linalg.norm(neighbors - position, axis=1))

                dof = len(neighbors) - 1  # Degrees of freedom
                t_value = t.ppf(0.975, dof)  # Two-tailed t-value for 95% confidence interval
                ci_low = mean_distance - (t_value * std_distance / np.sqrt(len(neighbors)))
                ci_high = mean_distance + (t_value * std_distance / np.sqrt(len(neighbors)))

                pruned_neighbors = np.array([neighbor for neighbor in neighbors if ci_low <= np.linalg.norm(neighbor - position) <= ci_high])

                if len(pruned_neighbors) == 0:
                    continue

                mean_spacing = np.mean(np.linalg.norm(pruned_neighbors - position, axis=1))

                mean_spacing_frame.append(mean_spacing)

                for j in range(k):
                    x_neighbors_frame[j].append(neighbors[j, 0])
                    y_neighbors_frame[j].append(neighbors[j, 1])

            mean_spacing_list.extend(mean_spacing_frame)
            for j in range(k):
                if len(x_neighbors_frame[j]) < len(df):
                    avg_neighbor_x = np.mean(x_neighbors_frame[j])
                    x_neighbors_frame[j].extend([avg_neighbor_x] * (len(df) - len(x_neighbors_frame[j])))

                if len(y_neighbors_frame[j]) < len(df):
                    avg_neighbor_y = np.mean(y_neighbors_frame[j])
                    y_neighbors_frame[j].extend([avg_neighbor_y] * (len(df) - len(y_neighbors_frame[j])))

                df[x_neighbor_columns[j]] = x_neighbors_frame[j]
                df[y_neighbor_columns[j]] = y_neighbors_frame[j]

    if len(mean_spacing_list) < len(df):
        mean_spacing_list.extend([np.nan] * (len(df) - len(mean_spacing_list)))

    df['mean_spacing'] = mean_spacing_list

    return df


def generate_dataset(df, name):
    """
    Generates a dataset based on the DataFrame and bottleneck name.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing the pedestrian data.
        name (str): Name of the bottleneck.

    Returns:
        pd.DataFrame: Generated dataset.
    """
    # Generate the dataset according to your requirements
    dataset = df  # Placeholder, modify as needed
    dataset['bottleneck'] = name
    return dataset
###Weidmann model###
def weidmann_model(x, v0, l, T):
    return v0 * (1 - np.exp((l - x) / v0 / T))