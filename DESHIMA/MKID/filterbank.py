import math
import time
from progress.bar import Bar
t = time.time()

from scipy import interpolate
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib import rcParams
rcParams['font.family'] = 'monospace'
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from matplotlib import rc
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = 'C:/FFmpeg/bin/ffmpeg.exe'
rc('text', usetex=True)

import numpy as np

import os
import sys
sys.path.insert(1, '../')
import use_desim
sys.path.insert(1, '../desim')
import minidesim as dsm
sys.path.insert(1, r'../../Animation/')
# import SubplotAnimationTime as aniT
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

    def __init__(self, F_min, R, F_max = None, num_filters = 1):
        self.F_min = F_min
        if F_max == None:
            self.F_max = F_min
        else:
            self.F_max = F_max
        self.F0 = F_min
        self.R = R
        self.num_filters = num_filters
        F = np.zeros(self.num_filters)
        for i in range(self.num_filters):
            F[i] = self.F_min * (1 + 1/self.R)**i
        self.filters = F
        self.FWHM = self.filters/R
        self.path_model = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    def drawLorentzian(self, x_width):
        """Plots a Lorentzian curve with num_bins bins over a width of x_width.

        Parameters
        ------------
        x_range: scalar
            Width over which the Lorentzian is plotted
            Unit: Hz
        """
        num_bins = 500
        x_plot = np.linspace(self.F0 - 1/2 * x_width, self.F0 + 1/2 * x_width, num_bins)
        y_plot = self.calcLorentzian(x_plot)
        plt.plot(x_plot, y_plot)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Response')
        plt.title('Lorenzian curve with a peak at '+ str(self.F0 * 1e-9) + ' GHz')
        plt.show()

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
        eta_atm = use_desim.get_eta_atm(self.filters, pwv, EL)
        return eta_atm

    def getPoints_TP_curve(self, EL_vector, pwv, progressbar):
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
        Tb_sky, psd_KID_desim = use_desim_instance.calcT_psd_P(self.eta_atm_df, self.F_highres, self.eta_atm_func_zenith, self.filters, EL_vector, self.num_filters, pwv, self.R, progressbar)
        delta_F = self.filters/self.R
        delta_F = delta_F.reshape([delta_F.shape[0], 1])
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
        self.bar = Bar('Progress', max=(2 * self.num_filters + 1000), suffix='%(percent)d%%') #1000 bins for Lorentzian curve
        for i in range(0, len(pwv_vector)):
            Pkid, Tb_sky = self.getPoints_TP_curve(EL_vector, pwv_vector[i], self.bar)
            # filename_Pkid = "C:/Users/Esmee/Documents/BEP/DESHIMA/Python/BEP/Data/Pkid/Pkid_for_pwv_" \
            # + str(pwv_vector[i]) + ".txt"
            # filename_Tb_sky = "C:/Users/Esmee/Documents/BEP/DESHIMA/Python/BEP/Data/Tb_sky/Tb_sky_for_pwv_" \
            # + str(pwv_vector[i]) + ".txt"
            filename_Pkid = self.path_model + '/Data/Pkid/Pkid_for_pwv_' + str(pwv_vector[i]) + '.txt'
            filename_Tb_sky = self.path_model + '/Data/Tb_sky/Tb_sky_for_pwv_' + str(pwv_vector[i]) + ".txt"
            np.savetxt(filename_Pkid, Pkid)
            np.savetxt(filename_Tb_sky, Tb_sky)
            Pkid = 0; Tb_sky = 0
        self.bar.finish()

    def save_etaF_data(self, pwv_vector, EL):
        """
        Saves values of the atmospheric transmission eta_atm, that are obtained by the 'getPoints_etaF_curve' method.
        """
        eta_atm = np.zeros([len(pwv_vector), len(self.filters)]) #num_filters can also be another (larger) numbers
        for k in range(0, len(pwv_vector)):
            eta_atm[k, :] = self.getPoints_etaF_curve(pwv_vector[k], EL)
        # filename_eta_atm = "C:/Users/Esmee/Documents/BEP/DESHIMA/Python/BEP/Data/eta_atm/eta_atm.txt"
        # filename_F= "C:/Users/Esmee/Documents/BEP/DESHIMA/Python/BEP/Data/F/F.txt"
        filename_eta_atm = self.path_model + 'Data/eta_atm/eta_atm.txt'
        filename_F = self.path_model + '/Data/F/F.txt'
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
            filename_Pkid = self.path_model + '/Data/Pkid/Pkid_for_pwv_' + str(pwv_vector[i]) + '.txt'
            filename_Tb_sky = self.path_model + '/Data/Tb_sky/Tb_sky_for_pwv_' + str(pwv_vector[i]) + ".txt"
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
        filename_eta_atm = self.path_model + 'Data/eta_atm/eta_atm.txt'
        filename_F = self.path_model + '/Data/F/F.txt'
        eta_atm = np.loadtxt(filename_eta_atm)
        F = np.loadtxt(filename_F)
        return eta_atm, F

    def make_animation(self, pwv_vector, EL_vector):
        """
        Makes an animation that compares the atmospheric transmission to the
        calibration between the KID power and the sky temperature, using the
        class SubplotAnimationTime. The animatin is shown in a new window and
        saved as a video and a folder of frames.
        """
        # obtain the data
        length_EL_vector = len(EL_vector)
        Tb_sky, Pkid = self.load_TP_data(pwv_vector, EL_vector)
        eta_atm, F = self.load_etaF_data()
        F_vector = F/(1e9) #in GHz
        # peak_indices = self.filtersit_TPpwv_curve(pwv_vector, EL_vector)
        SubplotAnimation_1 = aniT.SubplotAnimationTime(F_vector, eta_atm, Pkid, Tb_sky, pwv_vector)

        Writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1000)
        file_string = r'C:/Users/Esmee/Documents/BEP/DESHIMA/Animations/animation_tryout.mp4'
        SubplotAnimation_1.save(file_string, writer=Writer)
        plt.show()

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
            f = interpolate.SmoothBivariateSpline(EL_vector_long, Pkid_vector, \
            Tb_sky_vector, s = len(EL_vector_long))
            f_pwv = interpolate.SmoothBivariateSpline(Pkid_vector, EL_vector_long, \
            pwv_vector_long, s = len(Pkid_vector), kx = 3, ky = 3)
            # knots = 2
            # f_pwv = interpolate.LSQBivariateSpline(Pkid_vector, EL_vector_long, \
            # pwv_vector_long, tx = np.linspace(0, 1e-12, knots), ty = np.linspace(20., 90., knots), kx = 1, ky = 1)
            name = self.path_model + '\Data\splines_Tb_sky\spline_' + '%.1f' % (self.filters[j]/1e9) +'GHz'
            name_pwv = self.path_model + '\Data\splines_pwv\spline_' + '%.1f' % (self.filters[j]/1e9) +'GHz'
            np.save(name, np.array(f))
            np.save(name_pwv, np.array(f_pwv))
        return 0

    def check_pwv(self, EL, Pkid):
        pwv_vector = np.array([])
        for i in range(0, 200):
            f_load = np.load(self.path_model + '\Data\splines_pwv\spline_' \
            + '%.1f' % (self.filters[i]/1e9) +'GHz.npy')
            f_function = f_load.item()
            pwv = f_function(Pkid, EL)
            pwv_vector = np.append(pwv_vector, pwv)
        plt.plot(self.filters[0:200], pwv_vector, marker='.')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Precipitable water vapor (mm)')
        plt.show()

num_filters = 347
F_min = 220e9
R = 500
length_EL_vector = 25
EL_vector = np.linspace(20., 90., length_EL_vector)
filterbank_1 = filterbank(F_min, R, 440e9, num_filters)
pwv_vector = np.logspace(-1, 0.35, 25)
filterbank_1.fit_TPpwvEL_curve(pwv_vector, EL_vector)
filterbank_1.save_TP_data(EL_vector, pwv_vector)

##------------------------------------------------------------------------------
## Code that might be useful later
##------------------------------------------------------------------------------

# filterbank_1.save_etaF_data(pwv_vector, 90.)
# filterbank_1.make_animation(pwv_vector, EL_vector)
# filterbank_1.check_pwv(60., 0.6e-12)

# plot dips in atmospheric transmission curve
# plt.plot(F/1e9, eta_atm[0], c='dodgerblue')
# plt.scatter(F[peak_indices]/1e9, eta_atm[0, peak_indices], c=np.linspace(-2*np.pi, 2*np.pi, len(peak_indices)), marker='.')
# plt.title('Dips of the atmospheric transmission curve')
# plt.xlabel('Frequency (GHz)')
# plt.ylabel('$\eta$')
# plt.show()
