# import MKID.photon_noise as pn

# noise_1 = pn.photon_noise(10**-13, 300*(10**9), 380)
# noise_1.drawTimeSignal(2, 1)

# # plt.style.use('dark_background')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm #colormap
import Atmosphere.use_aris as use_aris
from scipy.ndimage.filters import gaussian_filter
import sys
sys.path.append('./GalaxySpectrum/')
import GalaxySpectrum.spectrumCreate as galaxy

# Galaxy part
frequency,spectrum=galaxy.giveSpectrumInclSLs(12,2)
plt.loglog(frequency,spectrum,color='blue',label='Redshift = 2')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Flux density [Jy]')
plt.tight_layout()
plt.show()
#Atmosphere part
#Get atmosphere matrix
# filename = 'sample00.dat'
# aris_1 = use_aris.use_ARIS(filename)
#
# #Filter it with a Gaussian
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
