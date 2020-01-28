import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
# plt.style.use('dark_background')

class bb_source(object):
    """
    Can calculate the power of Johnson-Nyquist radiation or an approximation of
    a Johnson-Nyquist curve using bins of equal size. Can also plot the approximation
    of the Johnson-Nyquist curve.

    Properties
    ------------
    F_min : scalar
        Minimal frequency of the astronomical signal.
        Unit: Hz
    F_min : scalar
        Maximal frequency of the astronomical signal.
        Unit: Hz
    T: scalar
        Temperature of the radiation.
        Unit: K
    spec_res: scalar
        Spectral resolution of the DESHIMA instrument. This is generally
        F/(Delta_F) ~ 380 for DESHIMA and F/(Delta_F) ~ 500 for DESHIMA 2.0
        Unit: Hz
    num_bins: integer
        Number of bins used in the approximation of the Johnson-Nyquist curve.
    bin_centres: vector
        Centres of the bins used in the approximation of the Johnson-Nyquist curve,
        equally spaced between F_min and F_max.
        Unit: Hz
    bin_sides: vector
        Sides of the bins used in the approximation of the Johnson-Nyquist curve.
        Unit: Hz
    bin_width: scalar
        Width of the bins used in the approximation fo the Johnson-Nyquist curve.
        Unit: Hz
    """

    h = 6.62607004e-34
    sampling_rate = 160
    k = 1.38064852e-23

    def __init__(self, F_min, F_max, num_bins, T, spec_res):
        self.F_min = float(F_min)
        self.F_max = float(F_max)
        self.T = float(T)
        self.spec_res = spec_res
        self.num_bins = num_bins
        self.bin_centres = np.zeros(self.num_bins)
        self.bin_sides = np.linspace(self.F_min, self.F_max, self.num_bins + 1)
        self.bin_width = (self.F_max - self.F_min)/self.num_bins

    def JN_rad(self, F):
        """Calculates the power of Johnson-Nyquist radiation when the frequency is given

        Returns
        ------------
        P: scalar
            Power of Johnson-Nyquist radiation
            Unit: W
        """
        P = bb_source.h * F / (math.exp(bb_source.h * F / (bb_source.k * self.T))-1)
        return P

    def approx_JN_curve(self):
        """Approximates a Johnson-Nyquist curve with equally spaced bins between
        F_min and F_max

        Returns
        ------------
        [bin_centres, P_bin_centres]
        bin_centres: vector
            Vector containing the frequencies for which the power has been calculated
            Unit: Hz
        P_bin_centres: vector
            Power corresponding to the frequency in bin_centres, calculated with
            the Johnson-Nyquist radiation formula
            Unit: W
        """
        for i in range(0, self.num_bins):
            self.bin_centres[i] = 0.5*(self.bin_sides[i] + self.bin_sides[i+1])
        self.JN_rad_vec = np.vectorize(self.JN_rad)
        P_bin_centres = self.JN_rad_vec(self.bin_centres) * self.bin_width #scaled with bin_width
        return [self.bin_centres, P_bin_centres]

    def plot_JN_curve(self):
        """Plots an approximation of the Johnson-Nyquist curve"""
        x = bb_source.h * self.approx_JN_curve()[0] / (bb_source.k * self.T)
        y = self.approx_JN_curve()[1]
        fig, ax = plt.subplots()
        plt.loglog(x, y)
        plt.xlabel('$log(hf/kT)$')
        plt.ylabel('$log(P)$ in $10^{-12} \cdot log(W)$')
        plt.yticks(y[0:self.num_bins:30], ["%.3f" % z for z in y[0:self.num_bins:30]*10**12])
        ax = plt.gca()
        plt.show()
