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
import sys
# plt.style.use('dark_background')
# plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=10)

import DESHIMA.desim.desim as dsm
import Atmosphere.use_aris as use_aris
import Telescope.telescope_transmission as tt
import DESHIMA.use_desim as use_desim
import DESHIMA.MKID.photon_noise as pn
import Source.bb_source as bbs

class signal_transmitter(object):
    "Class that transmits the signal through the telescope"

    sampling_rate = 160
    f_chop = 10 #Hz

    def __init__(self, F_min, F_max, num_bins, T, spec_res, R, time, num_filters = 1, F0 = 220e9):
        # F0 is now one frequency with default 220e9. When we start adding multiple filters it could become an array
        self.F_min = F_min
        self.F_max = F_max
        self.num_bins = num_bins
        self.T = T
        self.spec_res = spec_res
        self.F0 = F0
        self.R = R
        self.time = time
        self.num_filters = num_filters

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
            noise_signal = pn.photon_noise(self.P_bin_centres[i], self.bin_centres[i], self.spec_res)
            noise_signal.delta_F = bb_source.bin_width
            signal_matrix[i, :] = noise_signal.calcTimeSignalBoosted(self.time)[1]
            if i == 0:
                x = noise_signal.calcTimeSignalBoosted(self.time)[0]
        y = np.sum(signal_matrix, axis=0)
        return [x, y]

    def transmit_signal_DESIM(self):
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
        [self.bin_centres, self.psd_bin_centres, filters] = use_desim.D2goal_calc(self.F_min, self.F_max, \
        self.num_bins, self.num_filters, self.R)[1:4] #vector, frequency
        # self.psd_bin_centres = use_desim.D2goal_calc()[2] #matrix, psd_KID, outdated version??

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

    def transmit_signal_DESIM_multf_atm(self, windspeed, filename_atm_data, beam_radius):
        print('Hello')
        self.num_samples = int(self.time*self.sampling_rate)
        time_vector = np.linspace(0, self.time, self.num_samples)
        #Initialize the power matrix

        filters = np.linspace(self.F_min, self.F_max, self.num_filters)

        #Atmosphere
        aris_instance = use_aris.use_ARIS(filename_atm_data)
        tt_instance = tt.telescope_transmission()
        pwv_matrix_filtered = tt_instance.filter_with_Gaussian(aris_instance.pwv_matrix, beam_radius)
        eta_atm_df, F_highres = dsm.load_eta_atm()
        eta_atm_func_zenith = dsm.eta_atm_interp(eta_atm_df)

        start = time.time()
        inputs = range(self.num_samples)
        num_cores = multiprocessing.cpu_count()
        print(num_cores) #now 63 hardcoded instead of num_cores
        power_matrix = Parallel(n_jobs=32)(delayed(signal_transmitter.processInput)(i, self, pwv_matrix_filtered, time_vector[i], i, self.f_chop, windspeed, eta_atm_df, eta_atm_func_zenith, F_highres) for i in inputs)
        power_matrix_res = np.zeros([self.num_filters, self.num_samples])
        for j in range(self.num_samples):
            power_matrix_res[:, j] = power_matrix[j]
        # for i in range(0, self.num_samples):
        #     # Obtain pwv for time = time_vector[i] and the corresponding piece of sky
        #     pwv_value = use_aris.use_ARIS.obt_pwv(pwv_matrix_filtered, time_vector[i], windspeed)
        #     t1 = time.time()
        #     # Obtain data from DESIM with the right pwv
        #     [self.bin_centres, self.psd_bin_centres, filters] = use_desim.D2goal_calc(self.F_min, self.F_max, \
        #     self.num_bins, self.num_filters, self.R, pwv_value, eta_atm_df, F_highres, eta_atm_func_zenith)[1:4] #vector, frequency
        #     t2 = time.time()
        #     # Calculate the power from psd_KID
        #     self.P_bin_centres = self.psd_bin_centres * (self.bin_centres[1]-self.bin_centres[0]) #matrix, power
        #
        #     # Initialize signal_matrix
        #     # signal_matrix = np.zeros([self.num_filters, self.num_bins])
        #     # summed_signal_matrix = np.zeros(self.num_filters)
        #
        #     noise_signal = pn.photon_noise(self.P_bin_centres, self.bin_centres)
        #     noise_signal.delta_F = self.bin_centres[1]-self.bin_centres[0] #constant for each filter/time/bin
        #     signal_matrix = noise_signal.calcTimeSignalBoosted(self.time, 1)
        #     power_matrix_res[:, i] = np.sum(signal_matrix, axis=1)

        #     # old version - slow:
        #     #   for j in range(0, self.num_filters):
        #     #     for k in range(0, self.num_bins):
        #     #         noise_signal = pn.photon_noise(self.P_bin_centres[j, k], self.bin_centres[k], self.spec_res)
        #     #         noise_signal.delta_F = self.bin_centres[1]-self.bin_centres[0] #constant for each filter/time/bin
        #     #
        #     #         signal_matrix[j, k] = noise_signal.calcTimeSignalBoosted(self.time, 1)
        #     #     power_matrix[j, i] = np.sum(signal_matrix[j, :])
        #     t3 = time.time()
        #     print('t2-t1', t2-t1) #longest
        #     print('t3-t2', t3-t2)
        return [time_vector, power_matrix_res, filters, start] #x is the time, power_matrix is a stochastic signal of the power, filters are the frequencies of the filters

    def processInput(i, self, pwv_matrix_filtered, time, count, f_chop, windspeed, eta_atm_df, eta_atm_func_zenith, F_highres):
        pwv_value = use_aris.use_ARIS.obt_pwv(pwv_matrix_filtered, time, count, f_chop, windspeed)
        [self.bin_centres, self.psd_bin_centres, filters] = use_desim.D2goal_calc(self.F_min, self.F_max, \
        self.num_bins, self.num_filters, self.R, pwv_value, eta_atm_df, F_highres, eta_atm_func_zenith)[1:4]
        # Calculate the power from psd_KID
        self.P_bin_centres = self.psd_bin_centres * (self.bin_centres[1]-self.bin_centres[0]) #matrix, power

        noise_signal = pn.photon_noise(self.P_bin_centres, self.bin_centres)
        noise_signal.delta_F = self.bin_centres[1]-self.bin_centres[0] #constant for each filter/time/bin
        signal_matrix = noise_signal.calcTimeSignalBoosted(self.time, 1)
        power_matrix_column = np.sum(signal_matrix, axis=1)
        return power_matrix_column

    def convert_P_to_Tsky(self, power_matrix, filters):
        T_sky_matrix = np.zeros(power_matrix.shape)
        for i in range(0, self.num_filters):
            # name = r'C:\Users\Esmee\Documents\BEP\DESHIMA\Python\BEP\Data\splines_Tb_sky\spline_' \
            # + '%.1f' % (filters[i]/1e9) +'GHz.npy'
            # name = r'C:\Users\Esmee\Documents\BEP\DESHIMA\Python\BEP\Data\splines_Tb_sky\spline_' \
            # + "{0:.1f}".format(filters[i]/1e9) +'GHz.npy'
            name = r'C:\Users\sup-ehuijten\Documents\DESHIMA-model_18_12_19\Python\BEP\Data\splines_Tb_sky\spline_' \
             + "{0:.1f}".format(filters[i]/1e9) +'GHz.npy'
            f_load = np.load(name, allow_pickle= True)
            f_function = f_load.item()
            for j in range(0, power_matrix.shape[1]):
                T_sky_matrix[i, j] = f_function(90., power_matrix[i, j])
        return T_sky_matrix

    def draw_signal(self, useDESIM = 1, multipleFilters = 1, inclAtmosphere = 1, \
    filter = [1], windspeed = None, filename_atm_data = None, beam_radius = None):
        plt.figure()
        # frequency = 220e9
        if useDESIM:
            if multipleFilters:
                # sgtitle("Mock-up time signal")
                if inclAtmosphere:
                    [x, power_matrix, filters, start] = self.transmit_signal_DESIM_multf_atm(windspeed, filename_atm_data, beam_radius)
                else:
                    [x, power_matrix, filters] = self.transmit_signal_DESIM_multf()
                T_sky_matrix = self.convert_P_to_Tsky(power_matrix, filters)
                num_plots = len(filter)
                # fig = plt.figure()
                # fig.set_dpi(200)
                # fig.set_size_inches(0.9, 0.9)
                for i in range(0, num_plots):
                    plt.subplot(1, num_plots, i + 1)
                    y = T_sky_matrix[filter[i]-1, :]
                    plt.title('Filter ' + str(filter[i]))
                    plt.plot(x, y, linewidth=0.5, color='darkblue')
                    plt.ticklabel_format(useOffset=False)
                    plt.xlabel('Time (s)')
                    plt.ylabel('T_sky (K)')
                plt.tight_layout()
                end = time.time()
                print('Elapsed time: ', end - start)
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

# order of the arguments: F_min, F_max, num_bins, T, spec_res, R, time, num_filters, F0
windspeed = 10 #m/s
filename_atm_data = 'sample00.dat'
beam_radius = 5. #m
signal_transmitter_1 = signal_transmitter(220e9, 440e9, 1500, 275, 380, 500, 2, 350)
# print(signal_transmitter.transmit_signal_DESIM_multf2(signal_transmitter_1, windspeed, filename_atm_data, beam_radius))
signal_transmitter_1.draw_signal(1, 1, 1, [5, 250, 320], windspeed, filename_atm_data, beam_radius)
