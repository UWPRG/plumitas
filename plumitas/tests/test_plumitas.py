from __future__ import absolute_import, division, print_function
import os.path as op

import plumitas as plm

data_path = op.join(plm.__path__[0], 'data')


def test_pbmetad():
    """
    Testing function to convert of COLVAR file to pandas DataFrame.

    """
    colvar_files = op.join(data_path, "pbmetad/COLVAR")
    hills_files = op.join(data_path, "pbmetad/HILLS")
    plumed_file = op.join(data_path, "pbmetad/plumed.dat")

    pbmetad = plm.load_project(colvar_files, hills_files,
                               method='pbmetad')
    # these things shouldn't work
    pbmetad.free_energy_surface('phi', 'psi')
    pbmetad.weight_frames()

    # these things should work
    pbmetad.get_bias_params(plumed_file, bias_type='pbmetad')

    pbmetad.reconstruct_bias_potential()
    pbmetad.weight_frames()
    axis_1 = pbmetad.potential_of_mean_force(['phi', 'psi'])
    axis_2 = pbmetad.free_energy_surface('phi', 'psi',
                                         weight='weight')

    assert pbmetad.colvar['weight'] is not None
    assert axis_1
    assert axis_2


def test_metad():
    """
    Testing function to convert of COLVAR file to pandas DataFrame.

    """
    colvar_files = op.join(data_path, "metad/COLVAR")
    hills_files = op.join(data_path, "metad/HILLS")
    plumed_file = op.join(data_path, "metad/plumed.dat")

    metad = plm.load_project(colvar_files, hills_files,
                             method='metad')
    # these things shouldn't work
    metad.free_energy_surface('phi', 'psi')
    metad.weight_frames()

    # these things should work
    metad.get_bias_params(plumed_file, bias_type='metad')

    metad.reconstruct_bias_potential()
    metad.weight_frames()
    axis_1 = metad.potential_of_mean_force(['phi', 'psi'])
    axis_2 = metad.free_energy_surface('phi', 'psi',
                                       weight='weight')

    assert metad.colvar['weight'] is not None
    assert axis_1
    assert axis_2


def test_no_bias():
    colvar_files = op.join(data_path, "pbmetad/COLVAR")
    hills_files = op.join(data_path, "pbmetad/HILLS")
    project = plm.load_project(colvar_files, hills_files)

    # need to add tests to confirm that these are parsed as expected
    bin_plumed = op.join(data_path, "plumed_bin.dat")
    bin_space_plumed = op.join(data_path,
                               "plumed_bin_space.dat")
    project.get_bias_params(bin_plumed, bias_type='pbmetad')
    project.get_bias_params(bin_space_plumed, bias_type='pbmetad')
    project.get_bias_params(bin_plumed, bias_type='metad')
    project.get_bias_params(bin_space_plumed, bias_type='metad')
