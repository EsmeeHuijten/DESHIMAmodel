#cython: language_level=3
import numpy as

def filter_response_function(F_bins_Lor_mesh, F_filters_mesh, R, eta_lens_antenna_rad, eta_circuit, psd_co, psd_jn_chip):
  # F_bins_Lor_mesh, F_filters_mesh = np.meshgrid(F_bins_Lor, F_filters)
  eta_circuit = calcLorentzian(F_bins_Lor_mesh, F_filters_mesh, R) * np.pi * F_filters_mesh/(2 * R) * 0.35
  eta_chip = eta_lens_antenna_rad * eta_circuit
  psd_KID = rad_trans(np.transpose(np.array(psd_co)), np.transpose(np.array(psd_jn_chip)), eta_chip)
  return psd_KID

def calcLorentzian(F_bins_Lor_mesh, F_filters_mesh, R):
  # F_bins_Lor_mesh, F_filters_mesh = np.meshgrid(F_bins_Lor, F_filters)
  FWHM = F_filters_mesh/R
  y_array = 1/np.pi * 1/2 * FWHM / ((F_bins_Lor_mesh-F_filters_mesh)**2 + (1/2 * FWHM)**2)
  return y_array

def rad_trans(rad_in, medium, eta):
  rad_out = eta * rad_in + (1 - eta) * medium
  return rad_out
