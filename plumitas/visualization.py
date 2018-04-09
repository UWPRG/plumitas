import numpy as np
from matplotlib import pyplot as plt


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
    if not weight:
        raise ValueError('You must supply frame weights to generate the FES.'
                         'Try using plumitas.get_frame_weights first.')

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
