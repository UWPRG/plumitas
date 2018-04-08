import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plumitas.core as core


class PBMetaDProject(core.SamplingProject):
    def __init__(self, colvar, hills, input_file=None,
                 bias_type='PBMetaD', multi=False):
        super(PBMetaDProject, self).__init__(colvar, hills,
                                             input_file=input_file,
                                             bias_type=bias_type,
                                             multi=multi)
        self.method = 'PBMetaD'

    def reconstruct_bias_potential(self):
        if not self.biased_CVs:
            print('self.biased_CVs not set.')
            return

        for CV in self.biased_CVs:
            if not self.biased_CVs[CV].sigma:
                print('ERROR: please set sigma and grid edges'
                      ' used to bias {}.'.format(CV))
                continue

            cv_tuple = self.biased_CVs[CV]
            sigma = cv_tuple.sigma
            grid_min = cv_tuple.grid_min
            grid_max = cv_tuple.grid_max
            periodic = False
            # check for angle
            if CV in self.periodic_CVs:
                periodic = True

            n_bins = 5 * (grid_max - grid_min) / sigma
            if ('grid_slicing' in self.bias_params.keys()
                    and 'grid_bin' in self.bias_params.keys()):
                bins = core.get_float(self.bias_params['grid_bin'])
                slicing = core.get_float(self.bias_params['slicing'])
                slice_bins = (grid_max - grid_min) / slicing
                n_bins = max(bins, slice_bins)
            elif ('grid_slicing' in self.bias_params.keys()
                  and 'grid_bin' not in self.bias_params.keys()):
                slicing = core.get_float(self.bias_params['slicing'])
                n_bins = (grid_max - grid_min) / slicing
            elif ('grid_bin' in self.bias_params.keys()
                  and 'grid_slicing' not in self.bias_params.keys()):
                n_bins = core.get_float(self.bias_params['grid_bin'])

            grid = np.linspace(grid_min, grid_max, num=n_bins)
            s_i = self.hills[CV][CV].values
            w_i = self.hills[CV]['height'].values

            # reshape for broadcasting
            s_i = s_i.reshape(len(s_i), 1)
            w_i = w_i.reshape(len(w_i), 1)

            hill_values = core.sum_hills(grid, s_i, sigma, periodic)
            bias_potential = sum(w_i * hill_values)

            self.static_bias[CV] = pd.Series(bias_potential,
                                             index=grid)
        return

    def weight_frames(self, temp=None):
        """
        Assign frame weights using the Torrie and Valleau reweighting
        method from a quasi-static bias potential. Adds a 'weight' column
        to self.colvar.

        Parameters
        ----------
        temp : float, None
            If self.temp exists, the user does not need to supply a temp
            because self.temp will take it's place anyway. If self.temp does
            not exist, temp must be supplied in the method call or an error
            will be printed with no furhter action.

        Returns
        -------
        None
        """
        if not self.static_bias:
            print('Torrie-Valleau reweighting requires a quasi static '
                  'bias funciton in each CV dimension. Please try '
                  'reconstruct_bias_potential before weight_frames.')
            return

        if self.temp:
            temp = core.get_float(self.temp[0])

        if not temp:
            print('Temp not parsed from PLUMED input file. ')

        k = 8.314e-3
        beta = 1 / (temp * k)

        bias_df = pd.DataFrame(index=self.colvar.index)
        for CV in self.static_bias.keys():
            cut_indices = pd.cut(self.colvar[CV].values,
                                 self.static_bias[CV].index,
                                 labels=self.static_bias[CV].index[1:])

            bias_df[CV] = np.exp(
                -self.static_bias[CV][cut_indices].values * beta
            )

        pb_potential = -np.log(np.sum(bias_df, axis=1)) / beta
        weight = np.exp(beta * pb_potential)

        self.colvar['weight'] = weight / np.sum(weight)
        return
