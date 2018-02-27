from __future__ import absolute_import, division, print_function
import os.path as op
import matplotlib.pyplot as plt

import plumitas as plm

data_path = op.join(plm.__path__[0], 'data')


def test_read_files():
    """
    Testing function to convert of COLVAR file to pandas DataFrame.

    """
    colvar_file = op.join(data_path, "mini_colvar")
    hills_file = op.join(data_path, "mini_hills")

    colvar_df = plm.read_colvar(colvar_file)
    hills_df = plm.read_hills(hills_file)

    assert colvar_df is not None
    assert hills_df is not None


def test_make_2d_free_energy_surface():
    """
    Testing function to convert of HILLS file to pandas DataFrame.

    """
    colvar_file = op.join(data_path, "mini_colvar")
    colvar_df = plm.read_colvar(colvar_file)

    axis = plm.make_2d_free_energy_surface(colvar_df,
                                           colvar_df.columns[1],
                                           colvar_df.columns[2])

    assert isinstance(axis, type(plt.gca()))
