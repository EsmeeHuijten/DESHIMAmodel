import numpy as np
import matplotlib.pyplot as plt
import scipy.special
# plt.style.use('dark_background')

#import sys
#sys.path.append('./DESHIMA/MKID/')

class photon_noise(object):
    """
    Can calculate different expressions of the NEP. It can draw a Poisson distribution.
    This class can also calculate a noisy time signal using a Gaussian estimate of the
    Poisson distribution with the NEP included in the standard deviation.

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
    delta_Al = 188e-6 * 1.602e-19
    eta = 0.4

    def __init__(self, power, frequency, delta_F, sampling_rate):
        self.power = power
        self.frequency = frequency
        self.delta_F = delta_F
        self.sampling_rate = sampling_rate

    def calcNEPsimple(self):
        """Calculates the simple NEP, with only the first term of the expression
        mentioned in  the 'First light' paper (DOI: 10.1038/s41550-019-0850-8)

        Returns
        ------------
        NEP_s: scalar or array
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
        NEP_b: scalar or array
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
        lambda_ = self.power/(photon_noise.h*self.frequency*self.sampling_rate)
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
        lambda_ = self.power/(photon_noise.h*self.frequency*self.sampling_rate)
        N = int(self.sampling_rate * time)
        x = np.linspace(0, time, N)
        y = np.random.poisson(lambda_, N)
        y_plot = y*(photon_noise.h*self.frequency)*self.sampling_rate

        # Calculate the standard deviation of the signal, averaged every 0.5 seconds
        i = 0
        j = 0
        y_means = np.zeros(2*time)
        while i<2*time:
            y_means[i] = np.mean(y_plot[j:j+int(0.5*self.sampling_rate-1)])
            j += int(0.5*self.sampling_rate)
            i += 1
        return [x, y_plot]

    def calcTimeSignalBoosted(self, time = 1, atm = 0):
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
        std_dev = self.calcNEPboosted() * np.sqrt(0.5*self.sampling_rate)
        if atm:
            delta_y = np.random.normal(0, std_dev, std_dev.shape)
            y = self.power + delta_y
            # y = self.power # to test without photon noise
            return y
        else:
            mean = self.power
            N = int(self.sampling_rate * time)
            x = np.linspace(0, time, N)
            y_plot = np.random.normal(mean, std_dev, N)

        return [x, y_plot]

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
