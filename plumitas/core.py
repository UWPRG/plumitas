import os
import re

import pandas as pd


##################################
#### READ PLUMED OUTPUT FILES ####
##################################


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


def parse_bias(filename='plumed.dat', method=None):
    """
    Function that takes experimental data and gives us the
    dependent/independent variables for analysis.

    Parameters
    ----------
    filename : string
        Name of the plumed input file used for enhanced sampling run.
    method : string
        Name of bias method used during
    Returns
    -------
    bias_args : dict
        Dictionary of key: value pairs from the plumed.dat file. Will
        facilitate automatic reading of parameter reading once
        core.SamplingProject class is implemented.
    """
    if not method:
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
    method = method.upper()
    bias_string = input_string.split(method)[1].lower()

    # use regex to create dictionary of arguments
    arguments = (re.findall(r'\w+=".+?"', bias_string)
                 + re.findall(r'\w+=[\S.]+', bias_string))

    # partition each match at '='
    arguments = [(m.split('=')[0], m.split('=')[1].split(','))
                 for m in arguments]
    bias_args = dict(arguments)

    return bias_args


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

    return


##################################
####  CORE PLUMITAS CLASSES   ####
##################################


class SamplingProject:
    def __init__(self, colvar, hills, multi):
        self.method = None
        self.colvar = read_colvar(colvar, multi)
        self.hills = read_hills(hills)
        self.traj = None
        self.biased_CVs = None
        self.static_bias = None


class MetaDProject(SamplingProject):
    pass


class PBMetaDProject(SamplingProject):
    pass

