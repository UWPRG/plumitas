import glob
import os
import re

import numpy as np
import pandas as pd

from collections import namedtuple

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


def sum_hills(grid_points, hill_centers, sigma):
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

    Returns
    -------
    bias_grid : ndarray
        Value of bias contributed by each hill at each grid point.
    """
    dist_from_center = grid_points - hill_centers
    bias_grid = np.exp(
        -(dist_from_center * dist_from_center) / (2 * sigma * sigma)
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

    raise KeyError('Sorry, the "{}" method is not yet supported.'
                   .format(method))


def get_float(string):
    """
    Silly helper function in case grid boundaries are pi.
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
    Base class for management and analysis of enhanced sampling project.
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

        self.bias_params = parse_bias(input_file, bias_type)
        self.biased_CVs = {CV: GridParameters(
            sigma=get_float(self.bias_params['sigma'][idx]),
            grid_min=get_float(self.bias_params['grid_min'][idx]),
            grid_max=get_float(self.bias_params['grid_max'][idx])
        )
            for idx, CV in enumerate(self.bias_params['arg'])
        }
        self.temp = self.bias_params['temp'][0]

    def reconstruct_bias_potential(self):
        if not self.biased_CVs:
            print('self.biased_CVs not set.')
            return

        for CV in self.biased_CVs:
            if not self.biased_CVs[CV].sigma:
                print('ERROR: please set sigma and grid edges'
                      ' used to bias {}.'.format(CV))
                continue

            sigma = self.biased_CVs[CV].sigma
            grid_min = self.biased_CVs[CV].grid_min
            grid_max = self.biased_CVs[CV].grid_max

            n_bins = 5 * (grid_max - grid_min) / sigma
            grid = np.linspace(grid_min, grid_max, num=n_bins)

            s_i = self.hills[CV][CV].values
            s_i = s_i.reshape(len(s_i), 1)
            w_i = self.hills[CV]['height'].values
            w_i = w_i.reshape(len(w_i), 1)

            hill_values = sum_hills(grid, s_i, sigma)
            bias_potential = sum(w_i * hill_values)

            self.static_bias[CV] = pd.Series(bias_potential,
                                             index=grid)

        # now combine if multidimensional bias was added
        # if len(self.biased_CVs) < 2:
        #     return
        #
        # k = 8.314e-3
        # beta = 1 / (self.temp * k)
        #
        # # TODO: add weight from all CVs based on values in self.colvar
        # pre_log_weight = 0  # should actually be array with d = # frames
        # for CV in self.biased_CVs:
        #     pre_log_weight += np.exp(-self.static_bias[CV] * beta)

        # for each time step: get value of each 1D bias potential.
        # add these based on the PBMetaD equation.

    # def get_frame_weights(self):
    #     if not self.static_bias:
    #         print('Missing self.static_bias. Please try '
    #               'self.reconstruct_bias_potential first.')
    #         return
    #
    #     w = np.exp(beta * df.loc[:, static_bias])
    #     df['weight'] = w / w.sum()
    #
    #     self.colvar['frame_weight'] =


class MetaDProject(SamplingProject):
    pass


class PBMetaDProject(SamplingProject):
    pass
