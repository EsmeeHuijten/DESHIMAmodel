"""
DESHIMA Model signal_transmitter

Transmits a signal through all components of the ASTE telescope,
which corresponds to transmitting a signal through multiple objects in the model
"""
#import time
import math
import matplotlib.pyplot as plt
#from matplotlib.ticker import StrMethodFormatter, NullFormatter
from joblib import Parallel, delayed
#import multiprocessing
import numpy as np
#import scipy.special
#import os
from pathlib import Path
#import sys
import galspec
# plt.style.use('dark_background')
# plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=10)

from .DESHIMA.desim import minidesim as dsm
from .Atmosphere import use_aris
from .Telescope import telescope_transmission as tt
from .DESHIMA import use_desim
from .DESHIMA.MKID import photon_noise as pn

def unwrap_processInput(st1, i, aris_instance, use_desim_instance, time_step, count):
    """
    Wrapper function for processInput, in order to avoid bugs with joblib.parallel
    """
    return st1.processInput(i=i, aris_instance=aris_instance, use_desim_instance=use_desim_instance, time_step=time_step, count=count)

class signal_transmitter(object):
    "Class that transmits the signal through all components of the model"

    sampling_rate = 160

    def __init__(self, input):
        self.input = input
        self.F_min = input['F_min']
        self.num_bins = input['num_bins']
        self.spec_res = input['spec_res']
        self.f_spacing = input['f_spacing']
        self.F0 = input['F_min']
        self.time = input['time']
        self.num_filters = input['num_filters']
        self.windspeed = input['windspeed']
        self.prefix_atm_data = input['prefix_atm_data']
        self.grid = input['grid']
        self.x_length_strip = input['x_length_strip']
        self.beam_radius = input['beam_radius']
        self.useDESIM = input['useDESIM']
        self.inclAtmosphere = input['inclAtmosphere']
        self.luminosity = input['luminosity']
        self.redshift = input['redshift']
        self.linewidth = input['linewidth']
        self.EL = input['EL']
        self.max_num_strips = input['max_num_strips']
        self.save_name_data = input['save_name_data']
        self.pwv_0 = input['pwv_0']
        self.D1 = input['D1']
        self.n_jobs = input['n_jobs']
        #if self.D1:
        #    print('Simulation of DESHIMA 1.0')
        #else:
        #    print('Simulation of DESHIMA 2.0')
        #self.path_model = os.path.dirname(os.path.abspath(__file__))
        self.path_model = Path(__file__).parent
        if input['savefolder'] == None:
            self.save_path = Path.cwd().joinpath('output_TiEMPO')
        else:
            folder = input['savefolder']
            self.save_path = Path(input['savefolder'])
            self.savepath = Path.cwd()
            while folder.startswith('.'):
                folder = folder.strip('.')
                folder = folder.strip('/')
                folder = folder.strip('\\')
                self.savepath = self.savepath.parent
            self.savepath = self.savepath.joinpath(folder)
        if Path.exists(self.save_path) == False:
            self.save_path.mkdir(parents = True)
        folder = input['sourcefolder']
        self.sourcepath = Path.cwd()
        while folder.startswith('.'):
            folder = folder.strip('.')
            folder = folder.strip('/')
            folder = folder.strip('\\')
            self.sourcepath = self.sourcepath.parent
        self.sourcepath = self.sourcepath.joinpath(folder)
        self.F_max = self.F_min * (1 + 1/self.f_spacing)**(self.num_filters - 1)
        F = np.logspace(np.log10(self.F_min), np.log10(self.F_max), self.num_filters)
        self.filters = F

    def save_filtered_pwv_map(self):
        """
        This function loads in all atmosphere strips, takes the part of them
        that is needed for the simulation, glues them together and filters them
        with a Gaussian filter. The file is saved in the './Data/output_ARIS/'
        directory with name filename.
        """
        aris_instance = use_aris.use_ARIS(self.sourcepath, self.prefix_atm_data, self.pwv_0,  self.grid, self.windspeed, self.time, 40)
        tt_instance = tt.telescope_transmission()
        aris_instance.filtered_pwv_matrix = tt_instance.filter_with_Gaussian(aris_instance.pwv_matrix, self.beam_radius)
        return aris_instance.dEPL_matrix, aris_instance.pwv_matrix, aris_instance.filtered_pwv_matrix

    def processInput(self, i, aris_instance, use_desim_instance, time_step, count):
        """
        This function gets the right values of the pwv and then transmits the signal
        through Desim (DESHIMA simulator). The psd that is obtained from Desim is
        integrated to obtain the power and finally, photon noise is added using
        a Gaussian estimation of the Poisson distribution.

        Returns
        ------------
        power_matrix_column: array
            Values of the power of the signal for one time sample. This matrix has shape
            [5 (number of different pwv values), number of filters]
            Unit: W
        """
        pwv_values = aris_instance.obt_pwv(time_step, count, self.windspeed)
        # pwv_values = np.ones([5, 1]) # to test without atmosphere fluctuations
        [results_desim, self.bin_centres, self.psd_bin_centres, filters] = use_desim_instance.transmit_through_DESHIMA(self, pwv_values)[0:4]

        # Calculate the power from psd_KID
        first_dif = self.bin_centres[1] - self.bin_centres[0]
        last_dif = self.bin_centres[-1] - self.bin_centres[-2]
        # delta_F = np.concatenate((np.array([0.]), np.logspace(np.log10(first_dif), np.log10(last_dif), self.num_bins-1)))
        delta_F = first_dif
        self.P_bin_centres = self.psd_bin_centres * delta_F #matrix, power
        noise_signal = pn.photon_noise(self.P_bin_centres, self.bin_centres, delta_F, self.sampling_rate)
        signal_matrix = noise_signal.calcTimeSignalBoosted(atm = 1)
        power_matrix_column = np.sum(signal_matrix, axis=2)
        return power_matrix_column

    def transmit_signal_DESIM_multf_atm(self):
        """
        This function transmits the signal through the final version of the
        DESHIMA model. It starts with calculating the number of samples needed
        and making a time vector. Then the atmosphere map from ARIS is loaded in
        and filtered. It then loads the galaxy spectrum and converts this to a
        psd. Finally, it gives all the needed information to the processInput
        function that calculates the signal with parallel computing.

        Returns
        ------------
        time_vector: vector
            The times at which the signal needs to be calculated
            Unit: s
        power_matrix_res: array
            Values of the power of the signal. This matrix has shape
            [5 (number of different pwv values), number of filters, number of time samples]
            Unit: W
        T_sky_matrix: array
            Values of the sky temperature of the signal. This matrix has shape
            [5 (number of different pwv values), number of filters, number of time samples]
            Unit: K
        self.filters: vector
            Center frequencies of the filters in the filterbank in the MKID chip
            Unit: Hz
        start: float
            The moment at which the calculations started, to be able to show the
            elapsed time after running the program.
            Unit: s
        """
        #start_non_parallel = time.time()
        self.num_samples = int(self.time*self.sampling_rate)
        time_vector = np.linspace(0, self.time, self.num_samples)

        #Atmosphere
        aris_instance = use_aris.use_ARIS(self.sourcepath,self.prefix_atm_data, self.pwv_0, self.grid, self.windspeed, self.time, self.max_num_strips, 0)
        tt_instance = tt.telescope_transmission()
        aris_instance.filtered_pwv_matrix = tt_instance.filter_with_Gaussian(aris_instance.pwv_matrix, self.beam_radius)
        # aris_instance = 0 # to test without atmosphere fluctuations

        self.eta_atm_df, self.F_highres = dsm.load_eta_atm()
        self.eta_atm_func_zenith = dsm.eta_atm_interp(self.eta_atm_df)

        #Galaxy
        self.frequency_gal, spectrum_gal = galspec.spectrum(self.luminosity, self.redshift, self.F_min/1e9, self.F_max/1e9, self.num_bins,self.linewidth, mollines = 'False')
        Ae = dsm.calc_eff_aper(self.frequency_gal*1e9, self.beam_radius) #1e9 added to convert the f to Hz
        self.psd_gal = spectrum_gal * Ae * 1e-26 * 0.5

        #DESHIMA
        use_desim_instance = use_desim.use_desim()

        #num_cores = multiprocessing.cpu_count()
        #end_non_parallel = time.time()
        #print('Elapsed time non-parallel part: ', end_non_parallel-start_non_parallel)
        #print('Going into parallel')
        #relpath =  'output_TiEMPO'
        #path_F = self.path_model.joinpath(relpath, self.save_name_data + "_F")
        #np.save(path_F, np.array(self.filters))
        #start = time.time()
        for l in range(0, 8, 1):
            step_round = math.floor(self.num_samples/8)
            inputs = range(l * step_round, (l+1) * step_round)
            power_matrix = Parallel(n_jobs=self.n_jobs,backend = 'threading')(delayed(unwrap_processInput)(self, i, aris_instance, use_desim_instance, time_vector[i], i) for i in inputs)
            power_matrix_res = np.zeros([power_matrix[0].shape[0], self.num_filters, step_round])
            for j in range(step_round):
                power_matrix_res[:, :, j] = power_matrix[j]
            T_sky_matrix = np.zeros([power_matrix[0].shape[0], self.num_filters, step_round])
            for k in range(power_matrix[0].shape[0]):
                    T_sky_matrix[k, :, :] = self.convert_P_to_Tsky(power_matrix_res[k], self.filters)
            path_T = self.save_path.joinpath(self.save_name_data + "_T_" + str(l))
            np.save(path_T, np.array(T_sky_matrix))
            path_P = self.save_path.joinpath(self.save_name_data + "_P_" + str(l))
            np.save(path_P, np.array(power_matrix_res))
            del T_sky_matrix, power_matrix, power_matrix_res
            #print('Finished round ' + str(l + 1) + ' out of 8')
        #end = time.time()
        #print('Elapsed time parallel part: ', end - start)
        return [time_vector, self.filters]

    def convert_P_to_Tsky(self, power_matrix, filters):
        """
        This function converts an array with power values to an array with sky
        temperature values, using the saved interpolation function from the
        '.Data/splines_Tb_sky/' repository.

        Returns
        ------------
        T_sky_matrix: array
            Values of the sky temperature that correspons to the values of the
            power that were passed to the function
            Unit: K
        """
        T_sky_matrix = np.zeros(power_matrix.shape)
        for i in range(0, self.num_filters):
            path = self.path_model
            if self.D1:
                filename = 'Data/splines_Tb_sky/spline_' + "{0:.1f}".format(filters[i]/1e9) +'GHz_D1.npy'
            else:
                filename = 'Data/splines_Tb_sky/spline_' + "{0:.1f}".format(filters[i]/1e9) +'GHz.npy'
            f_load = np.load(path.joinpath(filename), allow_pickle= True)
            f_function = f_load.item()
            for j in range(0, power_matrix.shape[1]):
                T_sky_matrix[i, j] = f_function(self.EL, power_matrix[i, j])
        return T_sky_matrix
"""
    def draw_signal(self, save_name_plot, plot_filters = [1]):
        """"""
        This function makes sure the signal is calculated for each filter and
        plots the signal for the filters given in plot_filters. It then saves
        plot in the same folder as this file with the name save_name_plot.

        The part of the signal that is plotted is the middle left position of
        the pwv values.
        """"""
        plt.figure()
        if self.useDESIM:
            if self.num_filters > 1:
                if self.inclAtmosphere:
                    [x, power_matrix, T_sky_matrix, filters, start] = self.transmit_signal_DESIM_multf_atm()
                else:
                    [x, power_matrix, filters] = self.transmit_signal_DESIM_multf()
                # T_sky_matrix = self.convert_P_to_Tsky(power_matrix[0], filters)
                num_plots = len(plot_filters)
                # fig.set_dpi(200)
                # fig.set_size_inches(0.9, 0.9)
                # plt.rcParams.update({'font.size': 22})
                for i in range(0, num_plots):
                    plt.subplot(1, num_plots, i + 1)
                    y = T_sky_matrix[0, plot_filters[i]-1, :]
                    plt.title('Filter ' + str(plot_filters[i]))
                    plt.plot(x, y, linewidth=0.5, color='darkblue')
                    plt.ticklabel_format(useOffset=False)
                    plt.xlabel('Time (s)')
                    plt.ylabel('T_sky (K)')
                # plt.tight_layout()
                #end = time.time()
                #print('Elapsed time parallel part: ', end - start)
                plt.savefig(save_name_plot, transparent=True)
                plt.show()
                return [x, power_matrix, T_sky_matrix, filters]
            else:
                [x, y] = self.transmit_signal_DESIM()
                plt.title('Mock-up time signal using DESIM')
        else:
            [x, y] = self.transmit_signal_simple()
            plt.title('Mock-up time signal using the spectral response of 1 filter')
        print('Standard deviation is ' + str(math.sqrt(np.var(y))))
        plt.plot(x, y)
        plt.ticklabel_format(useOffset=False)
        # ax.get_yaxis().get_major_formatter().set_useOffset(False)
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.show()
"""