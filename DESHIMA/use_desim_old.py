window_AR = True

import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import integrate

import sys
sys.path.append('./DESHIMA/desim/')
import minidesim as dsm
sys.path.append('./DESHIMA/')
import pyximport; pyximport.install()
# import Lorentzian

h = 6.62607004 * 10**-34
k = 1.38064852 * 10**-23
e = 1.60217662 * 10**-19 # electron charge
c = 299792458.
Delta_Al = 188 * 10**-6 * e # gap energy of Al
eta_pb = 0.4

#Esmee's function to calculate the Lorentzian
def calcLorentzian(F_bins_Lor_mesh, F_filters_mesh, R):
    # F_bins_Lor_mesh, F_filters_mesh = np.meshgrid(F_bins_Lor, F_filters)
    FWHM = F_filters_mesh/R
    y_array = 1/math.pi * 1/2 * FWHM / ((F_bins_Lor_mesh-F_filters_mesh)**2 + (1/2 * FWHM)**2)
    return y_array

def D2HPBW(F):
    HPBW = 29.*240./(F/1e9) * np.pi / 180. / 60. / 60.
    return HPBW

def eta_mb_ruze(F, LFlimit, sigma):
    '''F in Hz, LFlimit is the eta_mb at => 0 Hz, sigma in m'''
    eta_mb = LFlimit* np.exp(- (4.*np.pi* sigma * F/c)**2. )
    return eta_mb

def transmit_through_DESHIMA(signal_instance, pwv_value):
    # multiple filters
    F_min = signal_instance.F_min
    F_max = signal_instance.F_max
    num_filters = signal_instance.num_filters
    num_bins_Lor = signal_instance.num_bins
    R = signal_instance.R
    eta_atm_df = signal_instance.eta_atm_df
    F_highres = signal_instance.F_highres
    eta_atm_func_zenith = signal_instance.eta_atm_func_zenith
    psd_gal = signal_instance.psd_gal

    F_filters = np.linspace(F_min, F_max, num_filters)
    F_bins_Lor = np.linspace(F_min,F_max,num_bins_Lor)

    eta_lens_antenna_rad = 0.81

    HPBW = D2HPBW(F_bins_Lor)
    eta_mb = eta_mb_ruze(F=F_bins_Lor,LFlimit=0.8,sigma=37e-6) * 0.9 # see specs, 0.9 is from EM, ruze is from ASTE
    eta_circuit_simple = 0.35
    D2goal_input ={
        'eta_atm_df': eta_atm_df,
        'F_highres': F_highres,
        'eta_atm_func_zenith': eta_atm_func_zenith,
        'F' : F_bins_Lor,
        'pwv':pwv_value,
        'EL':60.,
        'R' : R,
        'eta_M1_spill' : 0.99,
        'eta_M2_spill' : 0.90,
        'n_wo_mirrors' : 4.,
        'window_AR' : window_AR,
        'eta_co' : 0.65, # product of co spillover, qo filter
        'eta_lens_antenna_rad' : eta_lens_antenna_rad, # D2_2V3.pdf, p14: front-to-back ratio 0.93 * reflection efficiency 0.9 * matching 0.98 * antenna spillover 0.993
        'eta_circuit' : eta_circuit_simple, # 'Alejandro Efficiency', from the feedpoint of the antenna to being absorbed in the KID.
        'eta_IBF' : 0.6,
        'KID_excess_noise_factor' : 1.1,
        'theta_maj' : HPBW,
        'theta_min' : HPBW,
        'eta_mb' : eta_mb,
        'Tb_cmb' : 2.725,
        'Tp_amb' : 273.,
        'Tp_cabin' : 290.,
        'Tp_co' : 4.,
        'Tp_chip' : 0.12,
        'snr' : 5,
        'obs_hours' :8.,
        'on_source_fraction':0.4*0.9, # ON-OFF 40%, calibration overhead of 10%
        'psd_gal': psd_gal
    }
    DESHIMA_transmitted = dsm.spectrometer_sensitivity(**D2goal_input) # takes a lot of time
    psd_co = DESHIMA_transmitted['psd_co'] #vector because of F
    psd_jn_chip = DESHIMA_transmitted['psd_jn_chip']

    F_bins_Lor_mesh, F_filters_mesh = np.meshgrid(F_bins_Lor, F_filters)

    #cython
    # psd_KID = Lorentzian.filter_response_function(F_bins_Lor_mesh, F_filters_mesh, R, eta_lens_antenna_rad, eta_circuit, psd_co, psd_jn_chip)

    # not cython
    eta_circuit = calcLorentzian(F_bins_Lor_mesh, F_filters_mesh, R) * math.pi * F_filters_mesh/(2 * R) * eta_circuit_simple
    eta_chip = eta_lens_antenna_rad * eta_circuit

    # calculate psd_KID with different values for pwv
    psd_medium = np.transpose((1-eta_chip)*np.transpose(np.array(psd_jn_chip)))
    psd_KID = np.zeros([psd_co.shape[1], num_filters, num_bins_Lor])
    for i in range(num_bins_Lor):
        psd_co_i = psd_co[i, :].reshape(psd_co[i, :].shape[0], 1)
        eta_chip_i = eta_chip[:, i].reshape(1, eta_chip[:, i].shape[0])
        psd_KID_in_i = eta_chip_i*psd_co_i
        result = psd_KID_in_i + psd_medium[i, :]
        psd_KID[:, :, i] = result

    return D2goal, F_bins_Lor, psd_KID, F_filters

def getpsd_KID():
    D2goal = D2goal_calc()[0]
    return D2goal['F'] ,D2goal['psd_KID']

def get_eta_atm(F, pwv, EL):
    eta_lens_antenna_rad = 0.81
    eta_circuit = 0.35
    data_names = ['eta_atm']
    eta_atm = np.zeros(len(F))
    for i in range(0, len(F)):
        eta_atm[i] = obt_data(F[i], pwv, EL, eta_lens_antenna_rad, eta_circuit, data_names)[0]
    return eta_atm

def obt_data(F, pwv, EL, eta_lens_antenna_rad, eta_circuit, data_names):

    HPBW = D2HPBW(F)
    eta_mb = eta_mb_ruze(F=F,LFlimit=0.8,sigma=37e-6) * 0.9 # see specs, 0.9 is from EM, ruze is from ASTE
    D2goal_input ={
        'F' : F,
        'pwv':pwv,
        'EL':EL,
        'R' : 0,
        'eta_M1_spill' : 0.99,
        'eta_M2_spill' : 0.90,
        'n_wo_mirrors' : 4.,
        'window_AR' : window_AR,
        'eta_co' : 0.65, # product of co spillover, qo filter
        'eta_lens_antenna_rad' : eta_lens_antenna_rad, # D2_2V3.pdf, p14: front-to-back ratio 0.93 * reflection efficiency 0.9 * matching 0.98 * antenna spillover 0.993
        'eta_circuit' : eta_circuit, # 'Alejandro Efficiency', from the feedpoint of the antenna to being absorbed in the KID.
        'eta_IBF' : 0.6,
        'KID_excess_noise_factor' : 1.1,
        'theta_maj' : HPBW,
        'theta_min' : HPBW,
        'eta_mb' : eta_mb,
        'Tb_cmb' : 2.725,
        'Tp_amb' : 273.,
        'Tp_cabin' : 290.,
        'Tp_co' : 4.,
        'Tp_chip' : 0.12,
        'snr' : 5,
        'obs_hours' :8.,
        'on_source_fraction':0.4*0.9 # ON-OFF 40%, calibration overhead of 10%
    }

    D2goal = dsm.spectrometer_sensitivity(**D2goal_input) # takes a lot of time
    data = []
    for el in data_names:
        data.append(np.array(D2goal[el]))
    return data

def calcT_psd_P(F1, F2, EL_vector, num_filters, pwv = 0.1, progressbar = None):
    eta_lens_antenna_rad = 0.81
    eta_circuit_simple = 0.35
    length_F_vector = 1000 #number of bins to calculate the Lorentzian
    length_EL_vector = len(EL_vector)

    F_filter = np.linspace(F1, F2, num_filters) #center frequencies of filters
    F = np.linspace(F1/2, F2*1.5, length_F_vector) #to calculate the Lorentzian

    # Initializing variables
    psd_KID = np.zeros([num_filters, length_F_vector, length_EL_vector])
    Tb_sky = np.zeros([num_filters, length_EL_vector])
    Pkid = np.zeros([num_filters, length_EL_vector])
    psd_co = np.zeros([length_F_vector, length_EL_vector])
    psd_jn_chip = np.zeros([length_F_vector, length_EL_vector])
    eta_circuit = np.zeros(length_F_vector)

    # Obtain psd_co and psd_jn_chip from desim
    for j in range(0, length_F_vector):
        F_bin= F[j]
        [psd_co[j, :], psd_jn_chip[j, :]] = obt_data(F_bin, pwv, EL_vector, \
        eta_lens_antenna_rad, eta_circuit_simple, ['psd_co', 'psd_jn_chip'])
        if progressbar:
            progressbar.next()

    # Obtain psd_kid
    for i in range(0, num_filters):
        # Putting a Lorentzian curve with peak height 0.35 and center frequency F_filter[i] in eta_circuit
        R = 500 #maybe make this non-hardcoded
        eta_circuit = calcLorentzian(F, F_filter[i], R) * math.pi * F_filter[i]/(2 * R) * 0.35
        eta_chip = eta_lens_antenna_rad * eta_circuit
        eta_chip_matrix = np.tile(eta_chip.reshape(len(eta_chip), 1), (1, length_EL_vector))
        psd_KID[i, :, :] =  dsm.rad_trans(psd_co, psd_jn_chip, eta_chip_matrix)
        if progressbar:
            progressbar.next()

    # Obtain Tb_sky and Pkid
    for k in range(0, num_filters):
        [Tb_sky[k, :], Pkid[k, :]]  = obt_data(F_filter[k], pwv, EL_vector, \
        eta_lens_antenna_rad, eta_circuit_simple, ['Tb_sky', 'Pkid'])
        if progressbar:
            progressbar.next()

    return Tb_sky, psd_KID, Pkid

# a = dsm.eta_atm_func(350.e9, 0.5, np.linspace(1, 90, 90))
# print(a.shape)
