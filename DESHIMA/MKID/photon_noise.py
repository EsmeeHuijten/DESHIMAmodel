import numpy as np
import matplotlib.pyplot as plt
import scipy.special
# plt.style.use('dark_background')
#     import pyximport; pyximport.install()
#     import timeSigBoost

import sys
sys.path.append('./DESHIMA/MKID/')
import pyximport; pyximport.install()
import timeSigBoost

class photon_noise(object):
    """
    Can calculate different expressions of the NEP. It can draw a Poisson distribution

    Properties
    ------------
    frequency : scalar
        Frequency of the astronomical signal.
        Unit: Hz
    power: scalar
        Power of the astronomical signal.
        Unit: W
    spec_res: scalar
        Spectral resolution of the DESHIMA instrument. This is generally F/(Delta_F) ~ 380 for DESHIMA1.0.
        Unit: Hz
    delta_F: scalar
        Frequency bandwidth
        Unit: -
    """

    h = 6.62607004e-34
    sampling_rate = 160
    delta_Al = 188e-6 * 1.602e-19
    eta = 0.4

    def __init__(self, power, frequency, spec_res = 380):
        self.power = power
        self.frequency = frequency
        self.spec_res = spec_res
        self.delta_F = self.frequency/spec_res

    def calcNEPsimple(self):
        """Calculates the simple NEP, with only the first term of the of the
        'First light' paper (DOI: 10.1038/s41550-019-0850-8)

        Returns
        ------------
        NEP_s: scalar
            Simple NEP
            Unit: W/Hz
        """
        self.NEP_s = np.sqrt(2*self.power*photon_noise.h*self.frequency)
        return self.NEP_s

    def calcNEPboosted(self):
        """Calculates the NEP, with only the complete expression of the NEP
        mentioned in the 'First light' paper (DOI: 10.1038/s41550-019-0850-8).

        Returns
        ------------
        NEP_b: scalar
            Complete (or 'boosted') NEP
            Unit: W/Hz
        """
        self.NEP_b = np.sqrt(2*self.power*(photon_noise.h * self.frequency + self.power/self.delta_F) \
        + 4 * photon_noise.delta_Al * self.power/photon_noise.eta)
        return self.NEP_b

    def drawPoissonDistr(self):
        """Calculates and plots a Poisson distribution, where the parameter of
        this distribution is given by lambda = power/(h*frequency*sampling_rate)
        """
        lambda_ = self.power/(photon_noise.h*self.frequency*photon_noise.sampling_rate)
        lambda_calc = lambda_ *10**-6
        x = np.linspace(0, 2*lambda_calc, 30)
        y = (lambda_calc**x) \
        /scipy.special.factorial(x) * np.exp(-lambda_calc)

        # plotting the Poisson distribution
        plt.ylabel('Probability')
        plt.xlabel('Number of photons per second')
        plt.title('The Poisson distribution with average n '+ "{:.2e}".format(lambda_))
        plt.plot(x, y)
        plt.show()

    def calcTimeSignalSimple(self, time):
        """Calculates a mock-up time signal of the photon noise using the simple NEP,
            by drawing samples from the Poisson distribution with parameter
            lambda = power/(h*frequency*sampling_rate)

        Parameters
        ------------
        time: scalar
            Time over which the mock-up time signal is made
            Unit: s

        Returns
        ------------
        [x, y_plot]
        x: vector
            Time vector with equally spaced intervals between t = 0 s and t = time s
            Unit: s
        y_plot: vector
            Power vector with numbers drawn from the Poisson distribution
        """
        lambda_ = self.power/(photon_noise.h*self.frequency*photon_noise.sampling_rate)
        N = int(photon_noise.sampling_rate * time)
        x = np.linspace(0, time, N)
        y = np.random.poisson(lambda_, N)
        y_plot = y*(photon_noise.h*self.frequency)*photon_noise.sampling_rate

        # Calculate the standard deviation of the signal, averaged every 0.5 seconds
        i = 0
        j = 0
        y_means = np.zeros(2*time)
        while i<2*time:
            y_means[i] = np.mean(y_plot[j:j+int(0.5*self.sampling_rate-1)])
            j += int(0.5*self.sampling_rate)
            i += 1
        # print('Standard deviation of the signal is', np.sqrt(np.var(y_means)))

        return [x, y_plot]

    def calcTimeSignalBoosted(self, time, atm = 0):
        """Calculates a mock-up time signal of the photon noise using the complete (or boosted) NEP,
            by drawing samples from the Poisson distribution with parameter
            lambda = NEP_s^2 * 1/2 * sampling_rate
        Parameters
        ------------
        time: scalar
            Time over which the mock-up time signal is made
            Unit: s

        Returns
        ------------
        [x, y_plot]
        x: vector
            Time vector with equally spaced intervals between t = 0 s and t = time s
            Unit: s
        y_plot: vector
            Power vector with numbers drawn from the Poisson distribution
        """
        # Make a mock-up time signal of the photon noise using the boosted NEP, by
        # drawing samples from the normal distribution
        #with cython
        return timeSigBoost.calcTimeSigBoost(self.power, self.frequency, self.delta_F, self.sampling_rate, atm)

        # without cython
        # std_dev = self.calcNEPboosted() * np.sqrt(0.5*self.sampling_rate)
        # if atm:
        #     delta_y = np.zeros(self.power.shape)
        #     num_filters = self.power.shape[0]
        #     for i in range(0, num_filters):
        #         delta_y[i, :] = np.random.normal(0, std_dev[i, :], len(std_dev[i, :]))
        #     y = self.power + delta_y
        #     return y
        # else:
        #     mean = self.power
        #     N = int(photon_noise.sampling_rate * time)
        #     x = np.linspace(0, time, N)
        #     y_plot = np.random.normal(mean, std_dev, N)
        #
        # return [x, y_plot]

    def drawTimeSignal(self, time, isBoosted):
        """Creates a plot of a mock-up time signal

        Parameters
        ------------
        time: scalar
            Time over which the mock-up time signal is made
            Unit: s
        isBoosted: boolean
            Whether the boosted expression of the NEP must be used (if False,
            the simple expression of the NEP is used)
        """
        if isBoosted:
            [x, y_plot] = self.calcTimeSignalBoosted(time)
            titleString = 'boosted model'
        else:
            [x, y_plot] = self.calcTimeSignalSimple(time)
            titleString = 'simple model'

        # Plotting a mock-up time signal
        plt.plot(x, y_plot, 'g')
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.title('Mock-up time signal of photon noise using the '+ titleString)
        plt.show()

# noise_1 = photon_noise(1e-13, 300e9, 380)
# noise_1.drawTimeSignal(2.0, 1)
