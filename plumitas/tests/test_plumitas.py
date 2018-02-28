from __future__ import absolute_import, division, print_function
import os.path as op

import plumitas as plm

data_path = op.join(plm.__path__[0], 'data')


def test_read_files():
    """
    Testing function to convert of COLVAR file to pandas DataFrame.

    """
    colvar_file = op.join(data_path, "COLVAR")
    hills_file = op.join(data_path, "HILLS")

    colvar_df = plm.read_colvar(colvar_file)
    hills_df = plm.read_hills(hills_file)

    assert colvar_df is not None
    assert hills_df is not None


def test_make_2d_free_energy_surface():
    """
    Testing function to convert of HILLS file to pandas DataFrame.

    """
    number_of_replicas = 4

    colvar_name = op.join(data_path, "COLVAR")
    single_colvar_df = plm.read_colvar(colvar_name, unbiased=True)
    multi_colvar_df = plm.read_colvar(colvar_name,
                                      multi=number_of_replicas,
                                      unbiased=True)

    axis = plm.make_2d_free_energy_surface(multi_colvar_df,
                                           multi_colvar_df.columns[0],
                                           multi_colvar_df.columns[1],
                                           temp=300)

    assert (len(single_colvar_df.iloc[:, 0])
            == len(multi_colvar_df.iloc[:, 0]) / number_of_replicas)
    assert axis


def test_potential_of_mean_force():
    """
    Testing function to convert of HILLS file to pandas DataFrame.

    """
    colvar_file = op.join(data_path, "COLVAR")
    colvar_df = plm.read_colvar(colvar_file)

    axis = plm.potential_of_mean_force(colvar_df,
                                       colvar_df.columns,
                                       temp=300)

    assert axis
