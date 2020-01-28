"""
DESHIMA Model signal_transmitter

Transmits a signal through all components of the ASTE telescope,
which corresponds to transmitting a signal through multiple objects in the model
"""
import time
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import scipy.special
import os
import sys
sys.path.append('./GalaxySpectrum/')
import GalaxySpectrum.spectrumCreate as galaxy
# plt.style.use('dark_background')
# plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=10)

import DESHIMA.desim.minidesim as dsm
import Atmosphere.use_aris as use_aris
import Telescope.telescope_transmission as tt
import DESHIMA.use_desim as use_desim
import DESHIMA.MKID.photon_noise as pn
import Source.bb_source as bbs

class signal_transmitter(object):
    "Class that transmits the signal through the telescope"

    sampling_rate = 160

    def __init__(self, input):
        self.input = input
        self.F_min = input['F_min']
        self.F_max = input['F_max']
        self.num_bins = input['num_bins']
        self.T = input['T']
        self.spec_res = input['spec_res']
        self.F0 = input['F_min']
        self.R = input['R']
        self.time = input['time']
        self.num_filters = input['num_filters']
        self.windspeed = input['windspeed']
        self.prefix_atm_data = input['prefix_atm_data']
        self.grid = input['grid']
        self.beam_radius = input['beam_radius']
        self.useDESIM = input['useDESIM']
        self.inclAtmosphere = input['inclAtmosphere']

    def transmit_signal_simple(self):
        bb_source = bbs.bb_source(self.F_min, self.F_max, self.num_bins, self.T, self.spec_res)
        [self.bin_centres, self.P_bin_centres] = bb_source.approx_JN_curve()
        filterbank = fb.filterbank(self.F0, self.R)
        self.filter_response = filterbank.calcLorentzian(self.bin_centres)

        self.P_bin_centres = self.P_bin_centres * self.filter_response * math.pi * self.F0 / (2 * self.R)
        # self.P_total = sum(self.P_bin_centres) * bb_source.bin_width
        # print(self.P_total)
        signal_matrix = np.zeros([self.num_bins, int(self.time*self.sampling_rate)])
        for i in range(0, self.num_bins):
            noise_signal = pn.photon_noise(self.P_bin_centres[i], self.bin_centres[i], self.sampling_rate, self.spec_res)
            noise_signal.delta_F = bb_source.bin_width
            signal_matrix[i, :] = noise_signal.calcTimeSignalBoosted(self.time)[1]
            if i == 0:
                x = noise_signal.calcTimeSignalBoosted(self.time)[0]
        y = np.sum(signal_matrix, axis=0)
        return [x, y]

    def transmit_signal_DESIM(self): #is not working atm
        """Makes mock-up time signal for 1 filter with 1 Lorentzian"""
        self.bin_centres, self.psd_bin_centres = use_desim.getpsd_KID()
        self.P_bin_centres = self.psd_bin_centres * (self.bin_centres[1]-self.bin_centres[0])
        print(self.bin_centres, self.P_bin_centres)
        print('total power ' + str((self.bin_centres[1]-self.bin_centres[0]) * np.sum(self.psd_bin_centres)))
        # plt.plot(self.bin_centres, self.P_bin_centres)
        # plt.show()
        self.num_bins = len(self.bin_centres)
        signal_matrix = np.zeros([self.num_bins, int(self.time*self.sampling_rate)])
        # print(signal_matrix.size)
        for i in range(0, self.num_bins):
            noise_signal = pn.photon_noise(self.P_bin_centres[i], self.bin_centres[i], self.spec_res)
            noise_signal.delta_F = self.bin_centres[1]-self.bin_centres[0]
            signal_matrix[i, :] = noise_signal.calcTimeSignalBoosted(self.time)[1]
            if i == 0:
                x = noise_signal.calcTimeSignalBoosted(self.time)[0]
        y = np.sum(signal_matrix, axis=0)
        return [x, y]

    def transmit_signal_DESIM_multf(self):
        # Obtain data from DESIM
        [self.bin_centres, self.psd_bin_centres, filters] = use_desim.transmit_through_DESHIMA(self.F_min, self.F_max, \
        self.num_bins, self.num_filters, self.R)[1:4] #vector, frequency

        # Calculate the power from psd_KID
        self.P_bin_centres = self.psd_bin_centres * (self.bin_centres[1]-self.bin_centres[0]) #matrix, power

        # Initialize signal_matrix
        self.num_filters = len(self.P_bin_centres[:, 0]) # compare to self.num_filters, redundant?
        self.num_bins = len(self.bin_centres) # compare to self.num_bins, redundant?
        self.num_samples = int(self.time*self.sampling_rate)
        signal_matrix = np.zeros([self.num_filters, self.num_bins, self.num_samples])
        summed_signal_matrix = np.zeros([self.num_filters, self.num_samples])
        # print(signal_matrix.size)
        for j in range(0, self.num_filters):
            for i in range(0, self.num_bins):
                noise_signal = pn.photon_noise(self.P_bin_centres[j, i], self.bin_centres[i], self.spec_res)
                noise_signal.delta_F = self.bin_centres[1]-self.bin_centres[0]
                signal_matrix[j, i, :] = noise_signal.calcTimeSignalBoosted(self.time)[1]
                if j == 0 and i == 0:
                    time_vector = noise_signal.calcTimeSignalBoosted(self.time)[0]
            summed_signal_matrix[j, :] = np.sum(signal_matrix[j, :, :], axis=0)
            power_matrix = summed_signal_matrix

        return [time_vector, power_matrix, filters] #x is the time, summed_signal_matrix is the power (stochastic signal)

    def save_filtered_pwv_map(self):
        # windspeed = 1320000.0/self.time
        windspeed = 10
        aris_instance = use_aris.use_ARIS(self.prefix_atm_data, self.grid, self.windspeed, self.time)
        tt_instance = tt.telescope_transmission()
        aris_instance.filtered_pwv_matrix = tt_instance.filter_with_Gaussian(aris_instance.pwv_matrix, self.beam_radius)
        path = os.path.dirname(os.path.abspath(__file__))
        filename = '/Data/output_ARIS/remove_me.txt'
        np.savetxt(path + filename, aris_instance.filtered_pwv_matrix)

    def transmit_signal_DESIM_multf_atm(self):
        self.num_samples = int(self.time*self.sampling_rate)
        time_vector = np.linspace(0, self.time, self.num_samples)
        #Initialize the power matrix

        filters = np.linspace(self.F_min, self.F_max, self.num_filters)

        #Atmosphere
        if self.windspeed*self.time > 655350.0:
            aris_instance = use_aris(self.prefix_atm_data, self.grid, self.windspeed, self.time, 1)
        else:
            aris_instance = use_aris.use_ARIS(self.prefix_atm_data, self.grid, self.windspeed, self.time, 0)
            tt_instance = tt.telescope_transmission()
            aris_instance.filtered_pwv_matrix = tt_instance.filter_with_Gaussian(aris_instance.pwv_matrix, self.beam_radius)

        self.eta_atm_df, self.F_highres = dsm.load_eta_atm()
        self.eta_atm_func_zenith = dsm.eta_atm_interp(self.eta_atm_df)

        #Galaxy
        self.frequency_gal, spectrum_gal =galaxy.giveSpectrumInclSLs(12,2)
        Ae = dsm.calc_eff_aper(self.frequency_gal, self.beam_radius)
        self.psd_gal = spectrum_gal * Ae * 1e-26 * 0.5

        #DESHIMA
        use_desim_instance = use_desim.use_desim()

        start = time.time()
        inputs = range(self.num_samples)
        num_cores = multiprocessing.cpu_count()
        power_matrix = Parallel(n_jobs=32)(delayed(signal_transmitter.processInput)(i, self, aris_instance, use_desim_instance, time_vector[i], i) for i in inputs)
        power_matrix_res = np.zeros([power_matrix[0].shape[0], self.num_filters, self.num_samples])
        for j in range(self.num_samples):
            power_matrix_res[:, :, j] = power_matrix[j]
        T_sky_matrix = np.zeros([power_matrix[0].shape[0], self.num_filters, self.num_samples])
        for k in range(power_matrix[0].shape[0]):
                T_sky_matrix[k, :, :] = self.convert_P_to_Tsky(power_matrix_res[k], filters)
        return [time_vector, power_matrix_res, T_sky_matrix, filters, start] #x is the time, power_matrix is a stochastic signal of the power, filters are the frequencies of the filters

    def processInput(i, self, aris_instance, use_desim_instance, time_step, count):
        pwv_value = aris_instance.obt_pwv(time_step, count, self.windspeed)
        [self.bin_centres, self.psd_bin_centres, filters] = use_desim_instance.transmit_through_DESHIMA(self, pwv_value)[1:4]

        # Calculate the power from psd_KID
        self.P_bin_centres = self.psd_bin_centres * (self.bin_centres[1]-self.bin_centres[0]) #matrix, power

        noise_signal = pn.photon_noise(self.P_bin_centres, self.bin_centres, self.sampling_rate, self.spec_res)
        signal_matrix = noise_signal.calcTimeSignalBoosted(atm = 1)
        power_matrix_column = np.sum(signal_matrix, axis=2)
        return power_matrix_column

    def convert_P_to_Tsky(self, power_matrix, filters):
        T_sky_matrix = np.zeros(power_matrix.shape)
        for i in range(0, self.num_filters):
            path = os.path.dirname(os.path.abspath(__file__))
            filename = '/Data/splines_Tb_sky/spline_' + "{0:.1f}".format(filters[i]/1e9) +'GHz.npy'
            f_load = np.load(path + filename, allow_pickle= True)
            f_function = f_load.item()
            for j in range(0, power_matrix.shape[1]):
                T_sky_matrix[i, j] = f_function(90., power_matrix[i, j])
        return T_sky_matrix

    def draw_signal(self, save_name_plot, plot_filters = [1]):
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
                end = time.time()
                print('Elapsed time: ', end - start)
                plt.savefig(save_name_plot, transparent=True)
                plt.show()
                return
            else:
                [x, y] = self.transmit_signal_DESIM()
                plt.title('Mock-up time signal using DESIM')
        else:
            [x, y] = self.transmit_signal_simple()
            plt.title('Mock-up time signal using the spectral response of 1 filter')
        print('Standard deviation is ' + str(math.sqrt(np.var(y))))
        plt.plot(x, y)
        plt.ticklabel_format(useOffset=False)
        # plt.ylim(1.8774e-10, 1.882e-10)
        # ax.get_yaxis().get_major_formatter().set_useOffset(False)
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.show()
