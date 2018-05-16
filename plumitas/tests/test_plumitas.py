from __future__ import absolute_import, division, print_function
import os.path as op

import plumitas as plm
import mdtraj as md

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


def test_input():
    conf = op.join(data_path, "topology/conf.gro")
    traj = md.load(conf)
    top = traj.top

    plumed = {'header': {'restart': False,
                         'wholemolecules': ['protein', 'resname ALA']},

              'groups': {'sidechain_com': {'com': 'sidechain and resname ALA'}
                         },

              'collective_variables': {
                  'phi': {'torsion': {'angle': 'phi',
                                      'resid': 2,
                                      'atoms': '',
                                      }
                          },
                  'psi': {'torsion': {'atoms': '7,9,15,17',
                                      'angle': 'psi',
                                      'resid': 2,
                                      },
                          },
              },

              'bias': {'pbmetad': {'label': 'pbmetad',
                                   'arg': 'phi,psi',
                                   'temp': '300',
                                   'pace': '500',
                                   'biasfactor': '15',
                                   'height': '1.2',
                                   'sigma': '0.35,0.35',
                                   'grid_min': '-pi,-pi',
                                   'grid_max': 'pi,pi',
                                   'grid_spacing': '0.1,0.1',
                                   'file': 'HILLS_phi,HILLS_psi'
                                   }
                       },

              'footer': {'print': {'stride': '500',
                                   'arg': 'phi,psi,pbmetad.bias',
                                   'file': 'COLVAR'
                                   },
                         }
              }

    plm.generate_input(top, **plumed)
