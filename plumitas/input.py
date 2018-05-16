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

    for key, group in groups.items():
        label = f'{key}: '
        action = ''.join(action for action in group.keys())
        query = group[action]
        atoms = topology.select(query)
        a_str = ','.join(str(x) for x in atoms)
        group_string = f'{label}{action.upper()} ATOMS={a_str}\n'
        string.append(group_string)

    string.append('\n')
    return ''.join(string)


def cvs_to_string(cvs, topology):
    string = []
    for key, value in cvs.items():
        cv_type = ''.join(key for key in value.keys())
        label = f'{key}: {cv_type.upper()} '
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
            atom_lookup = {
                'phi': '(residue {} and name C) '
                       'or (residue {} and name N CA C)'
                       ''.format((resid - 1), resid),
                'psi': '(residue {} and name N CA C) '
                       'or (residue {} and name N'
                       ''.format(resid, (resid + 1))
            }

            query = atom_lookup[value['torsion']['angle']]
            atoms = topology.select(query)
            a_str = ','.join(str(x) for x in atoms)
            string.append(f'{a_str}\n')

    string.append('\n')
    return ''.join(string)


def bias_to_string(bias):
    string = []
    method = ''.join(key for key in bias.keys())
    string.append(f'{method.upper()} ...\n')

    for key, value in bias[method].items():
        string.append(f'{key.upper()}={value}\n')

    string.append(f'... {method.upper()}\n')
    string.append('\n')
    return ''.join(string)


def footer_to_string(footer):
    string = []
    for action, arguments in footer.items():
        string.append(f'{action.upper()} ')
        for key, value in footer[action].items():
            string.append(f'{key.upper()}={value} ')
        string.append('\n')

    string.append('\n')
    return ''.join(string)


def generate_input(mdtraj_top, out_file='plumed.dat', **kwargs):
    """
    Converts an input dictionary object into a plumed run file.

    Parameters
    ----------
    mdtraj_top : mdtraj.traj.Topology
        A mdtraj Topology object generated from an input configuration
        for your system. This allows automated atom selection with
        simple VMD-style atom queries. This method is especially useful
        when dealing with many collective variables and/or atom groups.
    out_file : string
        Name of file to be used with enhanced sampling simulation
        in PLUMED. Default plumed.dat is common choice.
    **kwargs : dict
        Dictionary containing plumed input stored in header, groups,
        cvs, bias, and footer sections. Eventually, these dicts will
        be conveniently created with an interactive GUI.

    Returns
    -------
    None
    """
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

    with open(out_file, 'w') as f:
        f.write(plumed_input)
    return
