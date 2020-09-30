import math
from pathlib import Path
from scipy import interpolate

import numpy as np

import os
from .. import use_desim
from ..desim import minidesim as dsm

# import DESHIMA.use_desim as use_desim
# import DESHIMA.desim.minidesim as dsm
# import SubplotAnimationSlider as aniS
# plt.style.use('dark_background')

class filterbank(object):
    """
    Class that represents the filterbank in an MKID chip.

    Properties
    ------------
    Fmin : scalar
        Resonance frequency of the filter with the smallest resonance frequency
        Unit: Hz
    R: scalar
        FWHM * F, where FWHM stands for full width at half maximum
        Unit: -
    Fmax : scalar
        Resonance frequency of the filter with the largest resonance frequency
        Unit: Hz
    num_filters: scalar
        Number of filters in the filterbank of the MKID
        Unit: -
    """

    def __init__(self, F_min, R, num_filters = 1, f_spacing = 380, num_bins = 1500, D1 = 0):
        self.F_min = F_min
        self.F0 = F_min
        self.R = R
        self.num_filters = num_filters
        self.f_spacing = f_spacing
        self.num_bins = num_bins
        self.F_max = F_min * (1 + 1/f_spacing)**(num_filters - 1)
        F = np.logspace(np.log10(self.F_min), np.log10(self.F_max), num_filters)
        self.filters = F
        self.FWHM = self.filters/R
        self.D1 = D1
        self.path_model = Path(__file__).parent.parent.parent

    def calcLorentzian(self, x_array):
        """Calculates values of a Lorentzian curve.

        Parameters
        ------------
        x_array: vector
            Frequencies of which the corresponding value of the Lorentzian curve
            is calculated
            Unit: Hz

        Returns
        ------------
        y_array: vector
            Values of Lorentzian curve, calculated with the values given in x_array
            Unit: -
        """
        y_array = 1/math.pi * 1/2 * self.FWHM / ((x_array-self.F0)**2 + (1/2 * self.FWHM)**2)
        return y_array


    def getPoints_etaF_curve(self, pwv, EL):
        """Obtains values of the atmospheric transmission eta_atm from desim,
        with given values of the precipitable water vapor and elevation.

        Parameters
        ------------
        pwv: vector or scalar
            Values of the precipitable water vapor for which the atmospheric
            transmission is calculated.
            Unit: mm
        EL: vector or scalar
            Values of the elevation for which the atmospheric
            transmission is calculated.
            Unit: degrees

        Returns
        ------------
        eta_atm: vector or scalar
            Values of the atmospheric transmission, calculated with the given
            values of pwv and EL
            Unit: -
        """
        eta_atm = dsm.eta_atm_func(self.filters, pwv, EL)
        return eta_atm

    def getPoints_TP_curve(self, EL_vector, pwv):
        """Obtains values of the KID power Pkid_summed and the sky temperature Tb_sky from desim,
        with given values of the precipitable water vapor and elevation.

        Parameters
        ------------
        EL_vector: vector or scalar
            Values of the elevation for which the KID power and sky temperature
            are to be calculated.
            Unit: degrees
        pwv: vector or scalar
            Values of the precipitable water vapor for which the KID power and
            sky temperature are to be calculated.
            Unit: mm

        Returns
        ------------
        Pkid_summed: vector or scalar
            Values of the KID power, calculated with the given values of pwv and
            EL. The filter response of the filters in the filterbank of the KID
            is taken into account and is integrated to obtain the KID power.
            Unit: W
        Tb_sky: vector or scalar
            Values of the sky temperature, calculated with the given
            values of pwv and EL.
            Unit: K
        """
        use_desim_instance = use_desim.use_desim()
        self.eta_atm_df, self.F_highres = dsm.load_eta_atm()
        self.eta_atm_func_zenith = dsm.eta_atm_interp(self.eta_atm_df)
        
        Tb_sky, psd_KID_desim, F_bins = use_desim_instance.calcT_psd_P(self.eta_atm_df, self.F_highres, self.eta_atm_func_zenith, self.filters, EL_vector, self.num_filters, pwv, self.R, self.num_bins, self.D1)
        first_dif = F_bins[1] - F_bins[0]
        last_dif = F_bins[-1] - F_bins[-2]
        # delta_F = np.concatenate((np.array([0.]), np.logspace(np.log10(first_dif), np.log10(last_dif), self.num_bins-1)))
        # delta_F = delta_F.reshape([1, delta_F.shape[0]])
        delta_F = first_dif
        Pkid = np.zeros(psd_KID_desim.shape)
        for i in range(psd_KID_desim.shape[2]):
            Pkid[:, :, i] = psd_KID_desim[:, :, i] * delta_F
        length_EL_vector = len(EL_vector)
        Pkid_summed = np.zeros([self.num_filters, length_EL_vector])
        for j in range(0, self.num_filters):
            Pkid_summed[j, :] = np.sum(Pkid[j, :, :], axis=0)
        return Pkid_summed, Tb_sky

    def save_TP_data(self, EL_vector, pwv_vector):
        """
        Saves values of the KID power Pkid_summed and the sky temperature Tb_sky, that are obtained by the 'getPoints_TP_curve' method.
        """
        for i in range(0, len(pwv_vector)):
            Pkid, Tb_sky = self.getPoints_TP_curve(EL_vector, pwv_vector[i])
            # filename_Pkid = "C:/Users/Esmee/Documents/BEP/DESHIMA/Python/BEP/Data/Pkid/Pkid_for_pwv_" \
            # + str(pwv_vector[i]) + ".txt"
            # filename_Tb_sky = "C:/Users/Esmee/Documents/BEP/DESHIMA/Python/BEP/Data/Tb_sky/Tb_sky_for_pwv_" \
            # + str(pwv_vector[i]) + ".txt"
            self.path_model.joinpath('Data/Pkid/').mkdir(parents = True, exist_ok = True)
            self.path_model.joinpath('Data/Tb_sky/').mkdir(parents = True, exist_ok = True)
            if self.D1:
                filename_Pkid = self.path_model.joinpath('Data/Pkid/Pkid_for_pwv_' + str(pwv_vector[i]) + '_D1.txt')
                filename_Tb_sky = self.path_model.joinpath('Data/Tb_sky/Tb_sky_for_pwv_' + str(pwv_vector[i]) + "_D1.txt")
            else:
                filename_Pkid = self.path_model.joinpath('Data/Pkid/Pkid_for_pwv_' + str(pwv_vector[i]) + '.txt')
                filename_Tb_sky = self.path_model.joinpath('Data/Tb_sky/Tb_sky_for_pwv_' + str(pwv_vector[i]) + ".txt")
            np.savetxt(filename_Pkid, Pkid)
            np.savetxt(filename_Tb_sky, Tb_sky)
            Pkid = 0; Tb_sky = 0

    def save_etaF_data(self, pwv_vector, EL):
        """
        Saves values of the atmospheric transmission eta_atm, that are obtained by the 'getPoints_etaF_curve' method.
        """
        eta_atm = np.zeros([len(pwv_vector), len(self.filters)]) #num_filters can also be another (larger) numbers
        for k in range(0, len(pwv_vector)):
            eta_atm[k, :] = self.getPoints_etaF_curve(pwv_vector[k], EL)
        # filename_eta_atm = "C:/Users/Esmee/Documents/BEP/DESHIMA/Python/BEP/Data/eta_atm/eta_atm.txt"
        # filename_F= "C:/Users/Esmee/Documents/BEP/DESHIMA/Python/BEP/Data/F/F.txt"
        self.path_model.joinpath('Data/eta_atm/').mkdir(parents = True, exist_ok = True)
        self.path_model.joinpath('Data/F/').mkdir(parents = True, exist_ok = True)
        filename_eta_atm = self.path_model.joinpath('Data/eta_atm/eta_atm.txt')
        filename_F = self.path_model.joinpath('Data/F/F.txt')
        np.savetxt(filename_eta_atm, eta_atm)
        np.savetxt(filename_F, self.filters)

    def load_TP_data(self, pwv_vector, EL_vector):
        """
        Loads values of the KID power Pkid_summed and the sky temperature Tb_sky, that are saved by the 'save_TP_data' method.
        """
        length_EL_vector = len(EL_vector)
        Pkid = np.zeros([len(pwv_vector), len(self.filters), length_EL_vector])
        Tb_sky = np.zeros([len(pwv_vector), len(self.filters), length_EL_vector])
        for i in range(0, len(pwv_vector)):
            if self.D1:
                filename_Pkid = self.path_model.joinpath('Data/Pkid/Pkid_for_pwv_' + str(pwv_vector[i]) + '_D1.txt')
                filename_Tb_sky = self.path_model.joinpath('Data/Tb_sky/Tb_sky_for_pwv_' + str(pwv_vector[i]) + "_D1.txt")
            else:
                filename_Pkid = self.path_model.joinpath('Data/Pkid/Pkid_for_pwv_' + str(pwv_vector[i]) + '.txt')
                filename_Tb_sky = self.path_model.joinpath('Data/Tb_sky/Tb_sky_for_pwv_' + str(pwv_vector[i]) + ".txt")
            Pkid[i, :, :] = np.loadtxt(filename_Pkid)
            Tb_sky[i, :, :] = np.loadtxt(filename_Tb_sky)
            # Pkid[i, :, :] = np.loadtxt("C:/Users/Esmee/Documents/BEP/DESHIMA/Python/BEP/Data/Pkid/Pkid_for_pwv_" \
            # + str(pwv_vector[i]) + ".txt")
            # Tb_sky[i, :, :] = np.loadtxt("C:/Users/Esmee/Documents/BEP/DESHIMA/Python/BEP/Data/Tb_sky/Tb_sky_for_pwv_" \
            # + str(pwv_vector[i]) + ".txt")
        return Tb_sky, Pkid

    def load_etaF_data(self):
        """
        Saves values of the atmospheric transmission eta_atm, that are obtained by the 'save_etaF_data' method.
        """
        filename_eta_atm = self.path_model.joinpath('Data/eta_atm/eta_atm.txt')
        filename_F = self.path_model.joinpath('Data/F/F.txt')
        eta_atm = np.loadtxt(filename_eta_atm)
        F = np.loadtxt(filename_F)
        return eta_atm, F


    def fit_TPpwvEL_curve(self, pwv_vector, EL_vector):
        """
        Fits a curve that relates the elevation EL and the KID power Pkid to the
        sky temperature Tb_sky. A smooth bivariate spline or third order is used
        for the interpolation. A separate 2D function is made for each filter in
        the filterbank of the MKID chip and each function is saved in a separate
        file.

        Parameters
        ------------
        EL_vector: vector or scalar
            Values of the elevation for which the KID power and sky temperature
            are to be calculated.
            Unit: degrees
        pwv: vector or scalar
            Values of the precipitable water vapor for which the KID power and
            sky temperature are to be calculated.
            Unit: mm
        """
        length_EL_vector = len(EL_vector)
        # eta_atm, F = self.load_etaF_data()
        # peak_indices = find_peaks(eta_atm[0, :]*(-1))[0] #gives indices of peaks

        #obtain data
        Tb_sky, Pkid = self.load_TP_data(pwv_vector, EL_vector)

        # make vectors of matrices
        pwv_vector_long = np.array([])
        EL_vector_long = np.array([])
        for i in range(0, len(pwv_vector)):
            pwv_vector_long = np.append(pwv_vector_long, pwv_vector[i]*np.ones(length_EL_vector))
            EL_vector_long = np.append(EL_vector_long, EL_vector)
        # make interpolations
        for j in range(0, self.num_filters):
            split_Tb_sky = tuple(np.vsplit(Tb_sky[:, j, :], len(Tb_sky[:, 0])))
            Tb_sky_vector = np.hstack(split_Tb_sky)
            split_Pkid = tuple(np.vsplit(Pkid[:, j, :], len(Pkid[:, 0])))
            Pkid_vector = np.hstack(split_Pkid)
            # if j in peak_indices:
            EL_vector_long = EL_vector_long.reshape([1, EL_vector_long.size])
            f = interpolate.SmoothBivariateSpline(EL_vector_long, Pkid_vector, \
            Tb_sky_vector, s = len(EL_vector_long))
            # f_pwv = interpolate.SmoothBivariateSpline(Pkid_vector, EL_vector_long, \
            # pwv_vector_long, s = len(Pkid_vector), kx = 3, ky = 3)
            if self.D1:
                name = self.path_model.joinpath('Data\splines_Tb_sky\spline_' + '%.1f' % (self.filters[j]/1e9) +'GHz_D1')
            else:
                name = self.path_model.joinpath('Data\splines_Tb_sky\spline_' + '%.1f' % (self.filters[j]/1e9) +'GHz')
            # name_pwv = self.path_model + '\Data\splines_pwv\spline_' + '%.1f' % (self.filters[j]/1e9) +'GHz_D1'
            np.save(name, np.array(f))

            f_load = np.load(str(name) + '.npy', allow_pickle= True)
            f_function = f_load.item()
        return 0

