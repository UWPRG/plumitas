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
    Testing functions to reweight based on COLVAR from biased
    simulations and generate 2D free energy surface.

    """
    number_of_replicas = 4

    colvar_name = op.join(data_path, "COLVAR")
    single_colvar_df = plm.read_colvar(colvar_name, unbiased=True)
    # create colvar DataFrame with from multiple walkers
    multi_colvar_df = plm.read_colvar(colvar_name,
                                      multi=number_of_replicas,
                                      unbiased=True)
    # overwrite weight column above with real, normalized weights
    plm.get_frame_weights(multi_colvar_df, bias='pb.bias', temp=300)

    # send over to 2D FES method
    axis = plm.make_2d_free_energy_surface(multi_colvar_df,
                                           multi_colvar_df.columns[0],
                                           multi_colvar_df.columns[1],
                                           weight='weight',
                                           temp=300)

    assert (len(single_colvar_df.iloc[:, 0])
            == len(multi_colvar_df.iloc[:, 0]) / number_of_replicas)
    assert axis


def test_potential_of_mean_force():
    """
    Testing functions to reweight based on COLVAR from biased
    simulations and generate 1D potential of mean force.

    """
    colvar_file = op.join(data_path, "COLVAR")
    colvar_df = plm.read_colvar(colvar_file)
    # get normalized frame weights
    plm.get_frame_weights(colvar_df, bias='pb.bias', temp=300)
    axis = plm.potential_of_mean_force(colvar_df,
                                       colvar_df.columns,
                                       weight='weight',
                                       temp=300)

    assert axis
