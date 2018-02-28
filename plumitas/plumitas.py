import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["read_colvar", "read_hills", "make_2d_free_energy_surface",
           "potential_of_mean_force"]


def read_colvar(filename):
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
    full_path = os.path.abspath(filename)

    with open(full_path, 'r') as f:
        header = f.readline().strip().split(" ")[2:]

    df = pd.read_csv(filename, comment='#', names=header,
                     delimiter='\s+', index_col=0)
    return df


def read_hills(filename):
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


def make_2d_free_energy_surface(df, x, y, temp, weight='pb.bias', bins=20,
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
    w = np.exp(beta * df.loc[:, weight])
    normalized_weight = w / w.sum()

    x_data = df[x].values
    y_data = df[y].values
    hist_w = normalized_weight.values

    x_edges = np.linspace(x_data.min(), x_data.max(), bins)
    y_edges = np.linspace(y_data.min(), y_data.max(), bins)

    hist, x_edges, y_edges = np.histogram2d(x_data, y_data,
                                            bins=(x_edges, y_edges),
                                            weights=hist_w)
    hist = hist.T

    hist = -np.log(hist) / beta
    hist = np.nan_to_num(hist)

    high_hist = hist > 20
    hist[high_hist] = 20
    hist = hist - hist.min()

    ax = plt.contourf(x_edges[1:], y_edges[1:], hist)
    cbar = plt.colorbar()
    plt.clim(0, clim)
    plt.set_cmap('viridis')
    cbar.ax.set_ylabel('A [kJ/mol]')

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(x)
    plt.ylabel(y)

    return ax


def potential_of_mean_force(df, collective_variable,
                            temp, weight='pb.bias', bins=20,
                            xlim=None, ylim=None):
    k = 8.314e-3
    beta = 1 / (temp * k)
    w = np.exp(beta * df.loc[:, weight])
    normalized_weight = w / w.sum()

    hist_x = df[collective_variable].values
    hist_w = normalized_weight.values

    x, y = np.histogram(hist_x, bins=bins, weights=hist_w)

    x = -np.log(x) / beta
    x = np.nan_to_num(x)

    high_x = x > 50
    x[high_x] = 50
    x = x - x.min()

    ax = plt.plot(y[:-1], x)
    plt.set_cmap('viridis')

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(x)
    plt.ylabel('A [kJ/mol]')

    return ax
