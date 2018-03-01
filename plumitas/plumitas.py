import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["read_colvar", "read_hills", "make_2d_free_energy_surface",
           "get_frame_weights", "potential_of_mean_force"]


def read_colvar(filename='COLVAR', multi=0, unbiased=False):
    """
    Function that takes experimental data and gives us the
    dependent/independent variables for analysis.

    Parameters
    ----------
    filename : string
        Name of the COLVAR file to read in.
    multi : int
        Tells the method to read 1 or more COLVAR files. Default falsy
        value (0) means read only 1 file.
    unbiased : bool
        If True, adds a 'weight' column of all 1s.
    Returns
    -------
    df : Pandas DataFrame
        CVs and bias as columns, time as index.
    """
    full_path = os.path.abspath(filename)
    colvar_paths = [full_path]

    if multi:
        colvar_paths = []
        for i in range(0, multi):
            replica_path = os.path.abspath(filename + "." + str(i))
            colvar_paths.append(replica_path)

    with open(colvar_paths[0], 'r') as f:
        header = f.readline().strip().split(" ")[2:]

    frames = []
    for path in colvar_paths:
        df = pd.read_csv(path, comment='#', names=header,
                         delimiter='\s+', index_col=0)

        # provide default weight if simulation was unbiased
        if unbiased:
            df['weight'] = 1

        if multi:
            frames.append(df)

    if multi:
        df = pd.concat(frames, axis=0, join='outer', ignore_index=True)

    return df


def read_hills(filename='HILLS'):
    """
    Function that takes experimental data and gives us the
    dependent/independent variables for analysis.

    Parameters
    ----------
    filename : string
        Name of the COLVAR file to read in.

    Returns
    -------
    df : Pandas DataFrame
        CVs and bias as columns, time as index.
    """
    return read_colvar(filename)


def get_frame_weights(df, temp, bias):
    # calculate normalized weight
    k = 8.314e-3
    beta = 1 / (temp * k)
    w = np.exp(beta * df.loc[:, bias])
    df['weight'] = w / w.sum()
    return


def make_2d_free_energy_surface(df, x, y, temp, weight=None, bins=20,
                                clim=None, xlim=None, ylim=None):
    """
    Create a 2D FES from a COLVAR file with static 'pb.bias'. This function
    will be modularized and generalized, but I wanted to include something
    more exciting than reading colvar/hills files for the first PyPI cut.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame generated from Plumed COLVAR file. This DataFrame must
        have a column with static 'pb.bias' - most likely generated from
        `mdrun rerun` - and at two CVs.
    x : string
        Name of one of the CVs (column name from df).
    y : string
        Name of one of the CVs (column name from df).
    bins : int
        Number of bins in each dimension to segment histogram.
    temp : float
        Temperature of simulation which generated Plumed file.
    weight : str
        Name of static bias column.
    clim : int
        Maximum free energy (in kJ/mol) for color bar.
    xlim : tuple/list
        Limits for x axis in plot (i.e. [x_min, x_max]).
    ylim : tuple/list
        Limits for y axis in plot (i.e. [y_min, y_max]).

    Returns
    -------
    axes: matplotlib.AxesSubplot
    """
    k = 8.314e-3
    beta = 1 / (temp * k)

    # grab ndarray of values from df
    x_data = df[x].values
    y_data = df[y].values
    w_data = df[weight].values

    # create bin edges
    x_edges = np.linspace(x_data.min(), x_data.max(), bins)
    y_edges = np.linspace(y_data.min(), y_data.max(), bins)

    # create weighted histogram, with weights converted to free energy
    hist, x_edges, y_edges = np.histogram2d(x_data, y_data,
                                            bins=(x_edges, y_edges),
                                            weights=w_data)
    hist = hist.T
    hist = -np.log(hist) / beta
    hist = np.nan_to_num(hist)

    # shift minimum energy to 0
    high_hist = hist > 20
    hist[high_hist] = 20
    hist = hist - hist.min()

    # fresh AxesSubplot instance
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # plot
    plt.contourf(x_edges[1:], y_edges[1:], hist)
    cbar = plt.colorbar()
    plt.clim(0, clim)
    plt.set_cmap('viridis')
    cbar.ax.set_ylabel('A [kJ/mol]')
    # add axis limits (if supplied) and labels
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(x)
    plt.ylabel(y)

    return ax


def potential_of_mean_force(df, collective_variables,
                            temp, weight=None, bins=100,
                            xlim=None, ylim=None):
    """
    Create PMF plot for one or several collective variables.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame generated from Plumed COLVAR file. This DataFrame must
        have a column with static 'pb.bias' - most likely generated from
        `mdrun rerun` - and at two CVs.
    collective_variables : list
        List of CVs you'd like to plot. These should be supplied in the
        form of a list of column names, or an instance of pd.Index using
        df.columns
    temp : float
        Temperature of simulation which generated Plumed file.
    weight : str
        Name of static bias column.
    bins : int
        Number of bins in each dimension to segment histogram.
    xlim : tuple/list
        Limits for x axis in plot (i.e. [x_min, x_max]).
    ylim : tuple/list
        Limits for y axis in plot (i.e. [y_min, y_max]).

    Returns
    -------
    axes: matplotlib.AxesSubplot
    """
    k = 8.314e-3
    beta = 1 / (temp * k)

    # fresh AxesSubplot instance
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    w_data = df[weight].values

    # add lines for each CV
    for cv in collective_variables:
        # grab ndarray from df[cv]
        x_data = df[cv].values

        # create weighted histogram, with weights converted to free energy
        x, y = np.histogram(x_data, bins=bins, weights=w_data)
        x = -np.log(x) / beta
        x = np.nan_to_num(x)

        # shift minimum energy to 0
        high_x = x > 50
        x[high_x] = 50
        x = x - x.min()

        # plot
        plt.plot(y[:-1], x)
        # add axis limits (if supplied) and labels
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(collective_variables)
        plt.ylabel('A [kJ/mol]')

    return ax
