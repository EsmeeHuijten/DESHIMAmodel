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

# Properties atmosphere
pwv_0 = 1.0 #mm
windspeed = 10 #m/s
prefix_atm_data = 'sample00.dat-'
grid = 0.2 #m
x_length_strip = 32768.0
max_num_strips = 40 #increase number if there are more atmosphere strips

# Properties Galaxy
luminosity = 13.7
redshift = 4.43
linewidth = 600

def calcMaxObsTime(windspeed):
    """
    Calculates the maximum observing time, using the number of gridpoints in
    one atmosphere strip, the gridwidth, the number of atmosphere strips and the
    windspeed
    """
    # maximum time
    # every strip has 32768 x-values
    separation = 2*0.5663 #m (116.8 arcseconds)
    max_obs_time = (x_length_strip - 3 * separation) *max_num_strips*grid/windspeed
    return max_obs_time

# Observation
EL = 60.
max_obs_time = calcMaxObsTime(windspeed)
# obs_time = 2 #s

obs_time = max_obs_time
# obs_time = 0.2
if obs_time > max_obs_time:
    raise ValueError('obs_time must be smaller than max_obs_time: ', max_obs_time)
draw_filters = [250] #starts counting from 1
save_name_plot = 'tryout.png'
save_name_data = 'output_model'

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
    'x_length_strip': x_length_strip,
    'beam_radius': 5.,
    'useDESIM': 1,
    'inclAtmosphere': 1,
    'luminosity': luminosity,
    'redshift': redshift,
    'linewidth': linewidth,
    'EL': EL,
    'max_num_strips': max_num_strips,
    'save_name_data': save_name_data,
    'pwv_0': pwv_0
}

signal_transmitter_1 = st.signal_transmitter(input_dictionary)
# signal_transmitter_1.save_filtered_pwv_map()
[time_vector, center_freq] = signal_transmitter_1.transmit_signal_DESIM_multf_atm()
print('Finished')
##------------------------------------------------------------------------------
## Code that might be useful later
##------------------------------------------------------------------------------

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
