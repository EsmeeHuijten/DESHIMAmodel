# plt.style.use('dark_background')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm #colormap
import signal_transmitter as st
import Atmosphere.use_aris as use_aris
from scipy.ndimage.filters import gaussian_filter
import sys
sys.path.append('./GalaxySpectrum/')
import GalaxySpectrum.spectrumCreate as galaxy

obs_time = 0.03125 #s
windspeed = 10 #m/s
prefix_atm_data = 'sample00.dat-'
grid = 0.2 #m
save_name_plot = 'tryout.png'
input_dictionary = {
    'F_min': 220e9,
    'F_max': 440e9,
    'num_bins': 1500,
    'T': 275,
    'spec_res': 500,
    'R': 500,
    'time': obs_time,
    'num_filters': 350,
    'windspeed': windspeed,
    'prefix_atm_data': prefix_atm_data,
    'grid': grid,
    'beam_radius': 5.,
    'useDESIM': 1,
    'inclAtmosphere': 1
}

signal_transmitter_1 = st.signal_transmitter(input_dictionary)
# signal_transmitter_1.save_filtered_pwv_map()
signal_transmitter_1.draw_signal(save_name_plot, [250])

# Galaxy test
# frequency,spectrum=galaxy.giveSpectrumInclSLs(12,2)
# plt.loglog(frequency,spectrum,color='blue',label='Redshift = 2')
# plt.xlabel('Frequency [GHz]')
# plt.ylabel('Flux density [Jy]')
# plt.tight_layout()
# plt.show()
#Atmosphere test
#Get atmosphere matrix
# filename = 'sample00.dat'
# aris_1 = use_aris.use_ARIS(filename)
#
#Gaussian filter test
# beam_radius = 5.
# std = np.sqrt(25/(2.0*np.log(10)))
# truncate = beam_radius/std
# output = gaussian_filter(aris_1.pwv_matrix, std, mode='mirror', truncate=truncate)
#
# #Plot the result
# fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
# axs[0].imshow(aris_1.pwv_matrix, cmap='viridis')
# axs[0].set_title('Before Gaussian filter')
# axs[1].imshow(output, cmap='viridis')
# axs[1].set_title('After Gaussian filter')
# # fig.suptitle('Categorical Plotting')
# plt.show()
