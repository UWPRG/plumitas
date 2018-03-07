import os

import pandas as pd


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
