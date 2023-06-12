# -*- coding: utf-8 -*-

import os
import pandas as pd
import plotly.graph_objects as go


def file_df_to_count_df(df,
                        ID_SUSCEPTIBLE=1,
                        ID_INFECTED=0,
                        ID_RECOVERED=2):
    """
    Converts the file DataFrame to a group count DataFrame that can be plotted.
    The ID_SUSCEPTIBLE and ID_INFECTED specify which ids the groups have in the Vadere processor file.
    """
    pedestrian_ids = df['pedestrianId'].unique()
    sim_times = df['simTime'].unique()
    group_counts = pd.DataFrame({'simTime': [], 'group-s': [], 'group-i': [], 'group-r': []})
    group_counts['simTime'] = sim_times
    group_counts['group-s'] = 0
    group_counts['group-i'] = 0
    group_counts['group-r'] = 0

    for pid in pedestrian_ids:
        simtime_group = df[df['pedestrianId'] == pid][['simTime', 'groupId-PID5']].values
        current_state = ID_SUSCEPTIBLE
        group_counts.loc[group_counts['simTime'] >= 0, 'group-s'] += 1
        for (st, g) in simtime_group:
            if g != current_state and g == ID_INFECTED and current_state == ID_SUSCEPTIBLE:
                current_state = g
                group_counts.loc[group_counts['simTime'] > st, 'group-s'] -= 1
                group_counts.loc[group_counts['simTime'] > st, 'group-i'] += 1
            elif g != current_state and g == ID_RECOVERED and current_state == ID_INFECTED:
                current_state = g
                group_counts.loc[group_counts['simTime'] > st, 'group-i'] -= 1
                group_counts.loc[group_counts['simTime'] > st, 'group-r'] += 1
                break
    return group_counts


def create_folder_data_scatter(folder):
    """
    Create scatter plot from folder data.
    :param folder:
    :return:
    """
    file_path = os.path.join(folder, "SIRinformation.csv")
    if not os.path.exists(file_path):
        return None
    data = pd.read_csv(file_path, delimiter=" ")

    print(data)

    ID_SUSCEPTIBLE = 1
    ID_INFECTED = 0
    ID_RECORVERED = 2

    group_counts = file_df_to_count_df(data, ID_INFECTED=ID_INFECTED, ID_SUSCEPTIBLE=ID_SUSCEPTIBLE,ID_RECOVERED=ID_RECORVERED)
    # group_counts.plot()
    scatter_s = go.Scatter(x=group_counts['simTime'],
                           y=group_counts['group-s'],
                           name='susceptible ' + os.path.basename(folder),
                           mode='lines')
    scatter_i = go.Scatter(x=group_counts['simTime'],
                           y=group_counts['group-i'],
                           name='infected ' + os.path.basename(folder),
                           mode='lines')
    scatter_r = go.Scatter(x=group_counts['simTime'],
                            y=group_counts['group-r'],
                            name='recovered ' + os.path.basename(folder),
                            mode='lines')
    return [scatter_s, scatter_i,scatter_r], group_counts
