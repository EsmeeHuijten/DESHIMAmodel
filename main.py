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

def calcMaxObsTime(windspeed):
    # maximum time
    # every strip has 32768 x-values
    max_obs_time = 32768.0*max_num_strips*grid/windspeed
    return max_obs_time

# Properties atmosphere
windspeed = 10 #m/s
prefix_atm_data = 'sample00.dat-'
grid = 0.2 #m
max_num_strips = 40 #increase number if there are more atmosphere strips

# Properties Galaxy
luminosity = 12
redshift = 2
linewidth = 300

# Observation
EL = 60.
max_obs_time = calcMaxObsTime(windspeed)
# obs_time = 0.03125 #s
obs_time = 1e14
if obs_time > max_obs_time:
    raise ValueError('obs_time must be smaller than max_obs_time: ', max_obs_time)
draw_filters = [250]
save_name_plot = 'tryout.png'

input_dictionary = {
    'F_min': 220e9,
    'num_bins': 1500,
    'T': 275,
    'spec_res': 500,
    'time': obs_time,
    'num_filters': 347,
    'windspeed': windspeed,
    'prefix_atm_data': prefix_atm_data,
    'grid': grid,
    'beam_radius': 5.,
    'useDESIM': 1,
    'inclAtmosphere': 1,
    'luminosity': luminosity,
    'redshift': redshift,
    'linewidth': linewidth,
    'EL': EL,
    'max_num_strips': max_num_strips
}

signal_transmitter_1 = st.signal_transmitter(input_dictionary)
# signal_transmitter_1.save_filtered_pwv_map()
signal_transmitter_1.draw_signal(save_name_plot, draw_filters)



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
