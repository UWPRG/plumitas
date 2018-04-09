import glob
import os
import re
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GridParameters = namedtuple('GridParameters',
                            ['sigma', 'grid_min', 'grid_max'])

"""
##################################
#### READ PLUMED OUTPUT FILES ####
##################################
"""


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
    # find all files matching filename
    all_hills = filename + '*'
    hills_names = glob.glob(all_hills)

    # parse each HILLS file with basic read_colvar call
    hills_frames = [read_colvar(hill_file)
                    for hill_file in hills_names]

    if len(hills_frames) == 1:
        return hills_frames[0]

    # return dictionary of HILLS dataframes with CV name as key
    return dict([(df.columns[0], df) for df in hills_frames])


def parse_bias(filename='plumed.dat', bias_type=None):
    """
    Function that takes experimental data and gives us the
    dependent/independent variables for analysis.

    Parameters
    ----------
    filename : string
        Name of the plumed input file used for enhanced sampling run.
    bias_type : string
        Name of bias method used during
    Returns
    -------
    bias_args : dict
        Dictionary of key: value pairs from the plumed.dat file. Will
        facilitate automatic reading of parameter reading once
        core.SamplingProject class is implemented.
    """
    if not filename:
        print('Bias parser requires filename. Please retry with '
              'valid filename.')
        return
    if not bias_type:
        print('Parser requires method to identify biased CVs. '
              'Please retry with valid method arg.')
        return

    # read input file into string
    full_path = os.path.abspath(filename)
    input_string = ''
    with open(full_path) as input_file:
        for line in input_file:
            input_string += line

    # isolate bias section
    bias_type = bias_type.upper()
    bias_string = input_string.split(bias_type)[1]

    # use regex to create dictionary of arguments
    arguments = (re.findall(r'\w+=".+?"', bias_string)
                 + re.findall(r'\w+=[\S.]+', bias_string))

    # partition each match at '='
    arguments = [(m.split('=')[0].lower(), m.split('=')[1].split(','))
                 for m in arguments]
    bias_args = dict(arguments)

    return bias_args


def sum_hills(grid_points, hill_centers, sigma, periodic=False):
    """
    Helper function for building static bias functions for
    SamplingProject and derived classes.

    Parameters
    ----------
    grid_points : ndarray
        Array of grid values at which bias potential should be
        calculated.
    hill_centers : ndarray
        Array of hill centers deposited at each bias stride.
    sigma : float
        Hill width for CV of interest.
    periodic : bool
        True if CV is periodic, otherwise False.

    Returns
    -------
    bias_grid : ndarray
        Value of bias contributed by each hill at each grid point.
    """
    dist_from_center = grid_points - hill_centers
    square = dist_from_center * dist_from_center

    if periodic:
        # can probably do something smarter than this!
        neg_dist = (np.abs(dist_from_center)
                    - (grid_points[-1] - grid_points[0]))
        neg_square = neg_dist * neg_dist
        square = np.minimum(square, neg_square)

    bias_grid = np.exp(
        -square / (2 * sigma * sigma)
    )
    return bias_grid


def load_project(colvar='COLVAR', hills='HILLS', method=None, **kwargs):
    """

    High-level function to read in all files associated with a Plumed
    enhanced sampling project. **kwargs supplied since different project
    types will be instantiated with different arguments.

    Parameters
    ----------
    colvar : string
        Name of the COLVAR file to read in.
    hills : string
        Name of the HILLS file to read in.
    method : string
        Name of enhanced sampling method used to bias the simulation.
        Supported methods will include "MetaD", "PBMetaD", and others.
        If the default None value is passed, plumitas will try to
        create

    Returns
    -------
    project : plumitas.SamplingProject
        Project base class, or subclass if 'method' is specified.
    """
    if not method:
        return SamplingProject(colvar, hills, **kwargs)

    if method.upper() == 'METAD':
        return MetaDProject(colvar, hills, **kwargs)

    if method.upper() == 'PBMETAD':
        return PBMetaDProject(colvar, hills, **kwargs)

    raise KeyError('Sorry, the "{}" method is not yet supported.'
                   .format(method))


def get_float(string):
    """
    Helper function in case grid boundaries are pi.

    Parameters
    ----------
    string : string
        Parameter string.

    Returns
    -------
    number : float
    """
    if string == 'pi':
        return np.pi
    elif string == '-pi':
        return -np.pi

    return float(string)


"""
###############################
#### CORE PLUMITAS CLASSES ####
###############################
"""


class SamplingProject:
    """
    Base class for management and analysis of an enhanced sampling project.
    """

    def __init__(self, colvar, hills, input_file=None,
                 bias_type=None, multi=False):
        self.method = None
        self.colvar = read_colvar(colvar, multi)
        self.hills = read_hills(hills)
        self.traj = None
        self.static_bias = {}

        if not input_file:
            return
        # if input file supplied, grab arguments from bias section
        self.bias_params = parse_bias(input_file, bias_type)
        self.biased_CVs = {CV: GridParameters(
            sigma=get_float(self.bias_params['sigma'][idx]),
            grid_min=get_float(self.bias_params['grid_min'][idx]),
            grid_max=get_float(self.bias_params['grid_max'][idx])
        )
            for idx, CV in enumerate(self.bias_params['arg'])
        }
        self.periodic_CVs = [CV for CV in self.biased_CVs
                             if self.biased_CVs[CV].grid_max == np.pi]
        if 'temp' in self.bias_params.keys():
            self.temp = get_float(self.bias_params['temp'][0])

    def get_bias_params(self, input_file, bias_type):
        """
        Method to grab bias parameters incase user forgot to supply
        plumed.dat or input file wasn't automatically identified in
        the working directory.

        Parameters
        ----------
        input_file : string
            Relative path to PLUMED input file. Most commonly called
            plumed.dat.
        bias_type : string
            String associated with biasing method used for enhanced
            sampling. Currently only "MetaD" and "PBMetaD" supported
            (case insensitive).

        Returns
        -------
        None

        """
        # if input file supplied, grab arguments from bias section
        self.bias_params = parse_bias(input_file, bias_type)
        self.biased_CVs = {CV: GridParameters(
            sigma=get_float(self.bias_params['sigma'][idx]),
            grid_min=get_float(self.bias_params['grid_min'][idx]),
            grid_max=get_float(self.bias_params['grid_max'][idx]),
        )
            for idx, CV in enumerate(self.bias_params['arg'])
        }
        self.periodic_CVs = [CV for CV in self.biased_CVs
                             if self.biased_CVs[CV].grid_max == np.pi]
        if 'temp' in self.bias_params.keys():
            self.temp = get_float(self.bias_params['temp'][0])

    def free_energy_surface(self, x, y, weight=None, bins=50,
                            clim=None, xlim=None, ylim=None,
                            energy_cut=50):
        """
        Create a 2D FES from a COLVAR file with frame weights.

        Parameters
        ----------
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
        energy_cut: float
            Cut off to exclude very high free energy values
            from histogram to help visualization.
        Returns
        -------
        axes: matplotlib.AxesSubplot
        """
        if not weight:
            print('You must supply frame weights to generate the FES. '
                  'Try using plumitas.get_frame_weights first.')
            return

        k = 8.314e-3
        beta = 1 / (self.temp * k)

        # grab ndarray of values from df
        x_data = self.colvar[x].values
        y_data = self.colvar[y].values
        w_data = self.colvar[weight].values

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
        high_hist = hist > energy_cut
        hist[high_hist] = energy_cut
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


class MetaDProject(SamplingProject):
    def __init__(self, colvar, hills, input_file=None,
                 bias_type='MetaD', multi=False):
        super(MetaDProject, self).__init__(colvar, hills,
                                           input_file=input_file,
                                           bias_type=bias_type,
                                           multi=multi)
        self.method = 'MetaD'

    def reconstruct_bias_potential(self):
        if not self.biased_CVs:
            print('self.biased_CVs not set.')
            return

        for idx, CV in enumerate(self.biased_CVs):
            if not self.biased_CVs[CV].sigma:
                print('ERROR: please set sigma and grid edges'
                      ' used to bias {}.'.format(CV))
                continue

            cv_tuple = self.biased_CVs[CV]
            sigma = cv_tuple.sigma
            grid_min = cv_tuple.grid_min
            grid_max = cv_tuple.grid_max

            periodic = False
            # check for angle
            if CV in self.periodic_CVs:
                periodic = True

            n_bins = 5 * (grid_max - grid_min) / sigma
            if ('grid_spacing' in self.bias_params.keys()
                    and 'grid_bin' in self.bias_params.keys()):
                bins = get_float(self.bias_params['grid_bin'][idx])
                slicing = get_float(self.bias_params['grid_spacing'][idx])
                slice_bins = (grid_max - grid_min) / slicing
                n_bins = max(bins, slice_bins)
            elif ('grid_spacing' in self.bias_params.keys()
                  and 'grid_bin' not in self.bias_params.keys()):
                slicing = get_float(self.bias_params['grid_spacing'][idx])
                n_bins = (grid_max - grid_min) / slicing
            elif ('grid_bin' in self.bias_params.keys()
                  and 'grid_spacing' not in self.bias_params.keys()):
                n_bins = get_float(self.bias_params['grid_bin'][idx])

            grid = np.linspace(grid_min, grid_max, num=n_bins)
            s_i = self.hills[CV].values

            s_i = s_i.reshape(len(s_i), 1)
            hill_values = sum_hills(grid, s_i, sigma, periodic)

            self.static_bias[CV] = pd.DataFrame(hill_values,
                                                columns=grid,
                                                index=self.hills[CV].index)

        return

    def weight_frames(self, temp=None):
        """
        Assign frame weights using the Torrie and Valleau reweighting
        method from a quasi-static bias potential. Adds a 'weight' column
        to self.colvar.

        Parameters
        ----------
        temp : float, None
            If self.temp exists, the user does not need to supply a temp
            because self.temp will take it's place anyway. If self.temp does
            not exist, temp must be supplied in the method call or an error
            will be printed with no furhter action.

        Returns
        -------
        None
        """
        if not self.static_bias:
            print('Torrie-Valleau reweighting requires a quasi static '
                  'bias funciton in each CV dimension. Please try '
                  'reconstruct_bias_potential before weight_frames.')
            return

        if self.temp:
            temp = get_float(self.temp)

        if not temp:
            print('Temp not parsed from PLUMED input file. ')

        k = 8.314e-3
        beta = 1 / (temp * k)

        bias_df = pd.DataFrame(columns=self.biased_CVs,
                               index=self.colvar.index)

        for CV in self.static_bias.keys():
            cut_indices = pd.cut(self.colvar[CV].values,
                                 self.static_bias[CV].columns,
                                 labels=self.static_bias[CV].columns[1:])

            bias_df[CV] = cut_indices
        test = bias_df.drop_duplicates()

        w_i = self.hills['height'].values

        for t, row in test.iterrows():
            weights = np.ones(len(self.hills))
            for CV in self.static_bias.keys():
                weights *= self.static_bias[CV][row[CV]].values

            static_bias = np.sum(w_i * weights)
            bias_df.loc[(bias_df['phi'] == row['phi'])
                        & (bias_df['psi'] == row['psi']),
                        'static_bias'] = static_bias

        weight = np.exp(beta * bias_df['static_bias'])
        self.colvar['weight'] = weight / np.sum(weight)
        return

    def potential_of_mean_force(self, collective_variables,
                                mintozero=True, xlabel='CV',
                                xlim=None, ylim=None):
        """
        Create PMF plot for one or several collective variables.

        Parameters
        ----------
        collective_variables : list
            List of CVs you'd like to plot. These should be supplied in
            the form of a list of column names, or an instance of
            pd.Index using df.columns
        mintozero : bool, True
            Determines whether or not to shift PMF so that the minimum
            is at zero.
        xlabel : string
            Label for the x axis.
        xlim : tuple/list
            Limits for x axis in plot (i.e. [x_min, x_max]).
        ylim : tuple/list
            Limits for y axis in plot (i.e. [y_min, y_max]).

        Returns
        -------
        axes: matplotlib.AxesSubplot
        """
        # fresh AxesSubplot instance
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        plt.xlabel(xlabel)
        plt.ylabel('A [kJ/mol]')

        w_i = self.hills['height'].values
        w_i = w_i.reshape(len(w_i), 1)

        # add lines for each CV
        for cv in collective_variables:
            hill_weights = w_i * self.static_bias[cv]

            static_bias = hill_weights.sum(axis=0)

            free_energy = (- static_bias
                           + static_bias.max())
            if not mintozero:
                free_energy = -static_bias

            plt.plot(free_energy)

        plt.xlim(xlim)
        plt.ylim(ylim)
        return ax


class PBMetaDProject(SamplingProject):
    def __init__(self, colvar, hills, input_file=None,
                 bias_type='PBMetaD', multi=False):
        super(PBMetaDProject, self).__init__(colvar, hills,
                                             input_file=input_file,
                                             bias_type=bias_type,
                                             multi=multi)
        self.method = 'PBMetaD'

    def reconstruct_bias_potential(self):
        if not self.biased_CVs:
            print('self.biased_CVs not set.')
            return

        for idx, CV in enumerate(self.biased_CVs):
            if not self.biased_CVs[CV].sigma:
                print('ERROR: please set sigma and grid edges'
                      ' used to bias {}.'.format(CV))
                continue

            cv_tuple = self.biased_CVs[CV]
            sigma = cv_tuple.sigma
            grid_min = cv_tuple.grid_min
            grid_max = cv_tuple.grid_max
            periodic = False
            # check for angle
            if CV in self.periodic_CVs:
                periodic = True

            n_bins = 5 * (grid_max - grid_min) / sigma
            if ('grid_spacing' in self.bias_params.keys()
                    and 'grid_bin' in self.bias_params.keys()):
                bins = get_float(self.bias_params['grid_bin'][idx])
                slicing = get_float(self.bias_params['grid_spacing'][idx])
                slice_bins = (grid_max - grid_min) / slicing
                n_bins = max(bins, slice_bins)
            elif ('grid_spacing' in self.bias_params.keys()
                  and 'grid_bin' not in self.bias_params.keys()):
                slicing = get_float(self.bias_params['grid_spacing'][idx])
                n_bins = (grid_max - grid_min) / slicing
            elif ('grid_bin' in self.bias_params.keys()
                  and 'grid_spacing' not in self.bias_params.keys()):
                n_bins = get_float(self.bias_params['grid_bin'][idx])

            grid = np.linspace(grid_min, grid_max, num=n_bins)
            s_i = self.hills[CV][CV].values
            w_i = self.hills[CV]['height'].values

            # reshape for broadcasting
            s_i = s_i.reshape(len(s_i), 1)
            w_i = w_i.reshape(len(w_i), 1)

            hill_values = sum_hills(grid, s_i, sigma, periodic)
            bias_potential = sum(w_i * hill_values)

            self.static_bias[CV] = pd.Series(bias_potential,
                                             index=grid)
        return

    def weight_frames(self, temp=None):
        """
        Assign frame weights using the Torrie and Valleau reweighting
        method from a quasi-static bias potential. Adds a 'weight' column
        to self.colvar.

        Parameters
        ----------
        temp : float, None
            If self.temp exists, the user does not need to supply a temp
            because self.temp will take it's place anyway. If self.temp does
            not exist, temp must be supplied in the method call or an error
            will be printed with no furhter action.

        Returns
        -------
        None
        """
        if not self.static_bias:
            print('Torrie-Valleau reweighting requires a quasi static '
                  'bias funciton in each CV dimension. Please try '
                  'reconstruct_bias_potential before weight_frames.')
            return

        if self.temp:
            temp = get_float(self.temp)
        if not temp:
            print('Temp not parsed from PLUMED input file. ')

        k = 8.314e-3
        beta = 1 / (temp * k)

        bias_df = pd.DataFrame(index=self.colvar.index)
        for CV in self.static_bias.keys():
            cut_indices = pd.cut(self.colvar[CV].values,
                                 self.static_bias[CV].index,
                                 labels=self.static_bias[CV].index[1:])

            bias_df[CV] = np.exp(
                -self.static_bias[CV][cut_indices].values * beta
            )

        pb_potential = -np.log(np.sum(bias_df, axis=1)) / beta
        weight = np.exp(beta * pb_potential)

        self.colvar['weight'] = weight / np.sum(weight)
        return

    def potential_of_mean_force(self, collective_variables,
                                mintozero=True, xlabel='CV',
                                xlim=None, ylim=None):
        """
        Create PMF plot for one or several collective variables.

        Parameters
        ----------
        collective_variables : list
            List of CVs you'd like to plot. These should be supplied in
            the form of a list of column names, or an instance of
            pd.Index using df.columns
        mintozero : bool, True
            Determines whether or not to shift PMF so that the minimum
            is at zero.
        xlabel : string
            Label for the x axis.
        xlim : tuple/list
            Limits for x axis in plot (i.e. [x_min, x_max]).
        ylim : tuple/list
            Limits for y axis in plot (i.e. [y_min, y_max]).

        Returns
        -------
        axes: matplotlib.AxesSubplot
        """
        # fresh AxesSubplot instance
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        plt.xlabel(xlabel)
        plt.ylabel('A [kJ/mol]')

        # add lines for each CV
        for cv in collective_variables:
            free_energy = (- self.static_bias[cv]
                           + self.static_bias[cv].max())
            if not mintozero:
                free_energy = -self.static_bias[cv]

            plt.plot(free_energy)

        plt.xlim(xlim)
        plt.ylim(ylim)
        return ax
