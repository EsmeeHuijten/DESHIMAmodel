import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from matplotlib import rc
rc('text', usetex=True)

import sys
sys.path.insert(1, 'C:/Users/Esmee/Documents/BEP/DESHIMA/Python/BEP/DESHIMA')
import calc_psd_KID as psd_KID

def make_etaF_plot(pwv_vector, EL_vector, F_vector):
    colors = ['darkblue', 'slateblue', 'dodgerblue']
    plt.figure()
    for i in range(0, len(pwv_vector)):
        eta_atm = psd_KID.get_eta_atm(F_vector, pwv_vector[i], EL_vector[i])
        plt.plot(F_vector/1e9, eta_atm, color=colors[i])
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Atmospheric transmission")
    # plt.legend(["pwv: " + str(pwv[0]) + " mm, EL: " + str(EL[0]) + "$^{\circ}$", \
    # "pwv: " + str(pwv[1]) + " mm, EL: " + str(EL[1]) + "$^{\circ}$", \
    #  "pwv: " + str(pwv[2]) + " mm, EL: " + str(EL[2]) + "$^{\circ}$"])
    # plt.title("Atmospheric transmission for different values of pwv and EL")
    plt.show()

def make_psdF_plot(pwv_vector, EL_vector, F_vector):
    colors = ['darkblue', 'slateblue', 'dodgerblue']
    eta_lens_antenna_rad = 0.81
    eta_circuit = 0.35
    R = 500
    data_names = ['psd_KID']
    plt.figure()
    for i in range(0, len(pwv_vector)):
        psd_KID_vector = np.zeros(len(F_vector))
        for j in range(0, len(F_vector)):
            data = psd_KID.obt_data(F_vector[j], pwv_vector[i], EL_vector[i], \
            eta_lens_antenna_rad, eta_circuit, data_names)
            psd_KID_vector[j] = data[0]
        plt.plot(F_vector/1e9, psd_KID_vector, color=colors[i])
    plt.xlabel("$F (GHz)$")
    plt.ylabel("$psd_{KID} (W/Hz)$")
    plt.legend(["pwv: " + str(pwv[0]) + " mm, EL: " + str(EL[0]) + "$^{\circ}$", \
    "pwv: " + str(pwv[1]) + " mm, EL: " + str(EL[1]) + "$^{\circ}$", \
     "pwv: " + str(pwv[2]) + " mm, EL: " + str(EL[2]) + "$^{\circ}$"])
    plt.title("Power spectral density of the KID signal for different values of pwv and EL")
    plt.show()

F = np.linspace(220e9, 440e9, 500)
# EL = np.linspace(20., 90., 3)
# pwv = np.linspace(0.1, 2.0, 3)
EL = [20., 60., 90.]
pwv = [0.1, 1.0, 2.5]
EL_1 = [90.]
pwv_1 = [1.0]
make_etaF_plot(pwv_1, EL_1, F)
