from __future__ import absolute_import, division, print_function
import os
import numpy as np
import pandas as pd
#from .due import due, Doi

__all__ = ["colvar_to_dataframe", "hills_to_dataframe"]


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
# due.cite(Doi("10.1167/13.9.30"),
#          description="Template project for small scientific Python projects",
#          tags=["reference-implementation"],
#          path='plumitas')


def colvar_to_dataframe(filename):
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

    pass


def hills_to_dataframe(filename):
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

    pass
