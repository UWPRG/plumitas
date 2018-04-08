import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plumitas.core as core


class MetaDProject(core.SamplingProject):
    def __init__(self, colvar, hills, input_file=None,
                 bias_type='MetaD', multi=False):
        super(MetaDProject, self).__init__(colvar, hills,
                                           input_file=input_file,
                                           bias_type=bias_type,
                                           multi=multi)
        self.method = 'MetaD'

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
            s_i = self.hills[CV].values

            s_i = s_i.reshape(len(s_i), 1)
            hill_values = core.sum_hills(grid, s_i, sigma, periodic)
            # bias_potential = sum(hill_values)/2.5

            self.static_bias[CV] = pd.DataFrame(hill_values,
                                                columns=grid,
                                                index=self.hills[CV].index)

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

        bias_df = pd.DataFrame(columns=self.biased_CVs,
                               index=self.colvar.index)

        for CV in self.static_bias.keys():
            cut_indices = pd.cut(self.colvar[CV].values,
                                 self.static_bias[CV].columns,
                                 labels=self.static_bias[CV].columns[1:])

            bias_df[CV] = cut_indices
        test = bias_df.drop_duplicates()

        w_i = self.hills['height'].values

        for t, row in test.iterrows():
            weights = np.ones(len(self.hills))
            for CV in self.static_bias.keys():
                weights *= self.static_bias[CV][row[CV]].values

            static_bias = np.sum(w_i * weights)
            bias_df.loc[(bias_df['phi'] == row['phi'])
                        & (bias_df['psi'] == row['psi']),
                        'static_bias'] = static_bias

        weight = np.exp(beta * bias_df['static_bias'])
        self.colvar['weight'] = weight / np.sum(weight)
        return

    def potential_of_mean_force(self, CV):
        w_i = self.hills['height'].values
        w_i = w_i.reshape(len(w_i), 1)
        hill_weights = w_i * -self.static_bias[CV]

        bias_potential = hill_weights.sum(axis=0)

        plt.plot(bias_potential)
        return
