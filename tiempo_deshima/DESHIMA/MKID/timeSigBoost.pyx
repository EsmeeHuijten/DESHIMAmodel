#cython: language_level=3
import numpy as np

h = 6.62607004e-34
delta_Al = 188e-6 * 1.602e-19
eta = 0.4

def calcTimeSigBoost(power, frequency, delta_F, sampling_rate, atm, time = 0):
  std_dev = calcNEPboosted(power, frequency, delta_F) * np.sqrt(0.5*sampling_rate)
  if atm:
      delta_y = np.zeros(power.shape)
      num_filters = power.shape[0]
      for i in range(0, num_filters):
          delta_y[i, :] = np.random.normal(0, std_dev[i, :], len(std_dev[i, :]))
      y = power + delta_y
      return y
  else:
      mean = power
      N = int(sampling_rate * time)
      x = np.linspace(0, time, N)
      y_plot = np.random.normal(mean, std_dev, N)
  return [x, y_plot]

def calcNEPboosted(power, frequency, delta_F):
  NEP = np.sqrt(2*power*(h * frequency + power/delta_F) \
      + 4 * delta_Al * power/eta)
  return NEP
