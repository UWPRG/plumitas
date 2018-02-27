import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from .due import due, Doi

__all__ = ["read_colvar", "read_hills", "make_2d_free_energy_surface"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
# due.cite(Doi("10.1167/13.9.30"),
#          description="Template project for small scientific Python projects",
#          tags=["reference-implementation"],
#          path='plumitas')


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


def make_2d_free_energy_surface(df, x_column, y_column, bins=20, beta=0.4,
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
    x_column : string
        Name of one of the CVs (column name from df).
    y_column : string
        Name of one of the CVs (column name from df).
    bins : int
        Number of bins in each dimension to segment histogram.
    beta : float
        1/(k_b * Temp)
    clim : int
        Maximum free energy (in kJ/mol) for color bar.
    xlim : list
        Limits for x axis in plot (i.e. [x_min, x_max]).
    ylim : list
        Limits for y axis in plot (i.e. [y_min, y_max]).

    Returns
    -------
    None
    """
    df['wt'] = np.exp(beta * df.loc[:, 'pb.bias'])
    df['normWt'] = df.loc[:, 'wt'] / df.loc[:, 'wt'].sum()

    x = df[x_column].values
    y = df[y_column].values
    w = df['normWt'].values

    xedges = np.linspace(x.min(), x.max(), bins)
    yedges = np.linspace(y.min(), y.max(), bins)

    hist, xedges, yedges = np.histogram2d(x, y,
                                          bins=(xedges, yedges),
                                          weights=w)
    hist = hist.T

    hist = -np.log(hist) / beta
    hist = np.nan_to_num(hist)

    high_hist = hist > 20
    hist[high_hist] = 20
    hist = hist - hist.min()

    plt.contourf(xedges[1:], yedges[1:], hist)
    cbar = plt.colorbar()
    plt.clim(0, clim)
    plt.set_cmap('viridis')
    cbar.ax.set_ylabel('weight [kJ/mol]')

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()
    return
