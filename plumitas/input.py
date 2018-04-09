import mdtraj as md


def header_to_string(header, topology):
    string = []
    if header['restart']:
        string.append('RESTART\n')

    if header['wholemolecules']:
        string.append('WHOLEMOLECULES ')
        for idx, query in enumerate(header['wholemolecules']):
            atoms = topology.select(query)
            a_str = ','.join(str(x) for x in atoms)
            entity = f'ENTITY{str(idx)}={a_str} '
            string.append(entity)
        string.append('\n')

    string.append('\n')
    return ''.join(string)


def groups_to_string(groups, topology):
    string = []

    string.append('\n')
    return ''.join(string)


def cvs_to_string(cvs, topology):
    string = []
    for key, value in cvs.items():
        cv_type = ''.join(key for key in value.keys())
        label = f'{key}: {cv_type.upper} '
        string.append(label)

        if cv_type == 'torsion':
            if value['torsion']['atoms']:
                atoms = value['torsion']['atoms']
                string.append(f'{atoms}\n')
                continue

            if (not value['torsion']['resid']
                    or not value['torsion']['resid']):
                print('If "atoms" are not specified, you must supply '
                      'both "resid" and "angle".')
                continue

            resid = value['torsion']['resid']
            atom_lookup = {'phi': f'(residue {resid - 1} and name C) '
                                  f'or (residue {resid} and name N CA C',
                           'psi': f'(residue {resid} and name N CA C) '
                                  f'or (residue {resid + 1} and name N',}

            query = atom_lookup(value['torsion']['angle'])
            atoms = topology.select(query)
            a_str = ','.join(str(x) for x in atoms)
            string.append(f'{a_str}\n')

    string.append('\n')
    return ''.join(string)


def bias_to_string(bias):
    string = []

    string.append('\n')
    return ''.join(string)


def footer_to_string(footer):
    string = []

    string.append('\n')
    return ''.join(string)


def generate_input(mdtraj_top, **kwargs):
    table, bonds = mdtraj_top.to_dataframe()
    plumed_dat = []
    if 'header' in kwargs.keys():
        plumed_dat.append(
            header_to_string(kwargs['header'], mdtraj_top)
        )
    if 'groups' in kwargs.keys():
        plumed_dat.append(
            groups_to_string(kwargs['groups'], mdtraj_top)
        )
    if 'collective_variables' in kwargs.keys():
        plumed_dat.append(
            cvs_to_string(kwargs['collective_variables'],
                          mdtraj_top)
        )
    if 'bias' in kwargs.keys():
        plumed_dat.append(
            bias_to_string(kwargs['bias'])
        )
    if 'footer' in kwargs.keys():
        plumed_dat.append(
            footer_to_string(kwargs['footer'])
        )

    plumed_input = ''.join(plumed_dat)
    # TODO: write output to file
    return
