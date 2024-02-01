import numpy as np
import pandas as pd
import os

DATA_PATH = "data/"


def read_station():
    station_40 = pd.read_csv(os.path.join(DATA_PATH, "station_40.csv"))
    station_49 = pd.read_csv(os.path.join(DATA_PATH, "station_49.csv"))
    station_63 = pd.read_csv(os.path.join(DATA_PATH, "station_63.csv"))
    station_80 = pd.read_csv(os.path.join(DATA_PATH, "station_80.csv"))
    return [station_49, station_80, station_40, station_63]


def sum_R(row):
    return (row['W_13'] + row['W_14'] + row['W_15']) / 3


def sum_T(row):
    return row[1:10].mean()


def compute_R_and_T(dfs):
    """
    Computes the average accumulated rainfall during summer (R) and the approximated cumulative Growing degree-day (T) across multiple datasets.

    Parameters:
    - dfs (list of pd.DataFrame): A list of pandas DataFrames, each containing rainfall and temperature data.

    Returns:
    - pd.DataFrame: A DataFrame containing the calculated average values of rainfall (R) and temperature (T)
      across all datasets. The DataFrame has two columns: 'R' for average rainfall and 'T' for average temperature.

    Example:
    >>> dfs = [df1, df2, df3]
    >>> compute_R_and_T(dfs)
           R         T
    0  1.524646  20.3342
    1  2.345675  21.1234
    ... 

    Notes:
    - This function assumes that all DataFrames in the input list (`dfs`) have the same length.
    """
    scenario_df = {'YEAR': range(len(dfs[0]))}
    for i, df in enumerate(dfs):
        scenario_df[f'R{i}'] = df.apply(sum_R, axis=1)
        scenario_df[f'T{i}'] = df.apply(sum_T, axis=1)

    scenario_df = pd.DataFrame(scenario_df)

    scenario_df['R'] = scenario_df[['R0', 'R1', 'R2', 'R3']].mean(axis=1)
    scenario_df['T'] = scenario_df[['T0', 'T1', 'T2', 'T3']].mean(axis=1)

    return scenario_df[['R', 'T']]


def subset_scenario(df: pd.DataFrame, i: int):
    """
    Filters a DataFrame based on the specified scenario index.

    This function takes a DataFrame containing 'R' and 'T' columns and filters it based on the scenario index.
    The scenario index determines the range of values for 'R' and 'T' to filter the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing 'R' and 'T' columns to be filtered.
    - i (int): The index of the scenario to apply for filtering.

    Returns:
    - pd.DataFrame: A DataFrame containing rows that satisfy the filtering conditions for the specified scenario.

    Example:
    >>> subset_scenario(df, 3)
       R     T
    0  1.5  20.5
    1  1.7  21.8
    ...
    """
    # Initialise the threshold
    R1 = [-np.inf, 1.8]
    R2 = [1.8, 2.2]
    R3 = [2.2, np.inf]
    T1 = [-np.inf, 21.2]
    T2 = [21.2, 22]
    T3 = [22, np.inf]

    # Dict of sets for each scenario
    scenario = {
        "1": {"R": R1, "T": T1},
        "2": {"R": R1, "T": T2},
        "3": {"R": R1, "T": T3},
        "4": {"R": R2, "T": T1},
        "5": {"R": R2, "T": T2},
        "6": {"R": R2, "T": T3},
        "7": {"R": R3, "T": T1},
        "8": {"R": R3, "T": T2},
        "9": {"R": R3, "T": T3},
    }

    Rmin, Rmax = scenario[f'{i}']['R']
    Tmin, Tmax = scenario[f'{i}']['T']

    # Create the mask
    selector = (df['R'] > Rmin) & (df['R'] <= Rmax) & (
        df['T'] > Tmin) & (df['T'] <= Tmax)

    filtered_df = df[selector]

    return filtered_df


def get_scenario(i: int):
    """
    Generates a scenario based on the given index.

    This function reads station data, computes a subset of the scenario based on criteria,
    and returns the filtered scenario.

    Parameters:
    - i (int): The index of the scenario to generate.

    Returns:
    - list of pandas.DataFrame: A list of pandas DataFrames representing the filtered scenario.

    Example:
    >>> get_scenario(3)
    [DataFrame1_filtered, DataFrame2_filtered, ...]
    """
    dfs = read_station()
    R_and_T = compute_R_and_T(dfs)
    scenario = subset_scenario(R_and_T, i)

    new_dfs = []
    for df in dfs:
        new_dfs.append(df.merge(scenario, how='inner',
                       left_index=True, right_index=True))
    return new_dfs
