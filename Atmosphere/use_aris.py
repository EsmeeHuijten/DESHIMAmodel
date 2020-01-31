import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = 'C:/FFmpeg/bin/ffmpeg.exe'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate, optimize
import os
import math
import numpy as np
import matplotlib.animation as animation
from matplotlib.transforms import Transform
from matplotlib.ticker import (
    AutoLocator, AutoMinorLocator)

import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
# from Atmosphere_model_Kah_Wuy.aris import load_aris_output
# import Telescope.telescope_transmission as tt

class use_ARIS(object):
    """
    This class is used to convert the atmosphere data that is obtained from
    ARIS to a useful datatype. It loads in the atmosphere data, converts it to
    a matrix and converts the Extra Path Length to the precipitable water vapor
    using the Smith-Weintraub equation for the Extra Path Length and the Ideal
    Gas Law (later, hopefully the VanDerWaals law).
    """

    a = 6.3003663 #m
    separation = 2*0.5663 #m (116.8 arcseconds)
    x_length_strip = 32768
    h = 1000 #m

    def __init__(self, prefix_filename, pwv_0, grid, windspeed, time, max_num_strips, loadCompletePwvMap = 0):
        #make function to convert EPL to pwv
        # pwv_vector = np.linspace(0, 5e-3, 1000)
        # e_vector = use_ARIS.calc_e_from_pwv(pwv_vector)
        # self.interp_e = interpolate.interp1d(e_vector, pwv_vector)
        self.pwv_0 = pwv_0
        self.prefix_filename = prefix_filename
        self.grid = grid
        self.windspeed = windspeed
        self.max_num_strips = max_num_strips
        self.path_model = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if __name__ == "__main__":
            self.pathname = "../Data/output_ARIS/"
        else:
            self.pathname = self.path_model + '/Data/output_ARIS/'
        self.filtered_pwv_matrix = "None"
        if loadCompletePwvMap:
            self.load_complete_pwv_map()
        else:
            self.initialize_pwv_matrix(time)

    def calc_e_from_pwv(pwv):
        """Calculates the simple partial pressure of water vapor, using the
        VanDerWaals equation of state and values for the parameters a and b
        taken from 'CRC Handbook for chemistry and physics', 64th edition.

        Returns
        ------------
        e: scalar or array
            partial pressure of water vapor
            Unit: kPa
        """
        a = 5.536e-4
        b = 3.049e-5
        T = 275
        rho = 55.4e3
        R = 8.314459
        e = R * T * pwv/(use_ARIS.h - b * pwv * rho) - (a * (pwv**2) * (rho**2)) / (use_ARIS.h**2)
        return e

    def calc_e_from_EPL(self, EPL):
        """Calculates the partial pressure of water vapor, using the Smith-Weintraub
        formula for Extra Path Length (DOI: 10.1109/JRPROC.1953.274297)

        Returns
        ------------
        e: scalar or array
            partial pressure of water vapor
            Unit: kPa
        """
        k2 = 70.4e2 #K/kPa
        k3 = 3.739e7 #K**2/kPa
        T = 275
        e = 1e6/self.h * EPL/(k2/T + k3/(T**2))
        return e

    def load_complete_pwv_map(self):
        """Loads the previously saved pwv map. This map is already filtered with
        a Gaussian beam.
        """
        print('Number of atmosphere strips loaded: ', self.max_num_strips)
        path = self.path_model + '/Data/output_ARIS/complete_filtered_pwv_map.txt'
        self.filtered_pwv_matrix = np.loadtxt(path)

    def initialize_pwv_matrix(self, time):
        """Initializes the pwv_matrix property of the use_ARIS instance. It loads
        in the amount of atmosphere strips that it needs, converts it into a matrix
        and then converts EPL to pwv.
        """
        max_distance = (time*self.windspeed + 2*self.separation)
        max_x_index = math.ceil(max_distance/self.grid)
        num_strips = min(math.ceil(max_x_index/self.x_length_strip), self.max_num_strips)
        print('Number of atmosphere strips loaded: ', num_strips)
        for i in range(num_strips):
            filename = self.prefix_filename + (3-len(str(i))) * "0" + str(i)
            d = np.loadtxt(self.pathname + filename, delimiter=',')
            nx = int(max(d[:, 0])) + 1
            ny = int(max(d[:, 1])) + 1
            epl= np.zeros([nx,ny])
            for j in range(len(d)):
                epl[int(d[j, 0]), int(d[j, 1])] = int(d[j, 2])
            if i == 0:
                self.dEPL_matrix = epl[:, 0:30]
            else:
                self.dEPL_matrix = np.concatenate((self.dEPL_matrix, epl[:, 0:30]), axis = 0)
        self.pwv_matrix = self.pwv_0 + (1/self.a * self.dEPL_matrix*1e-6)*1e3 #in mm
        # e_matrix = self.calc_e_from_EPL(self.dEPL_matrix*1e-3)
        # self.pwv_matrix = self.interp_e(e_matrix)*1e3 #in mm
        self.pwv_matrix = np.array(self.pwv_matrix)

    def obt_pwv(self, time, count, windspeed):
        """Obtains the precipitable Water vapor from the pwv_matrix at 5 different positions, to
        make sky chopping and nodding possible in 2 directons.

        Returns
        ------------
        pwv: array
            precipitable water vapor on at 5 different positions.
            Unit: m
        """
        pwv_matrix = self.filtered_pwv_matrix
        length_x = pwv_matrix.shape[0]
        positions = self.calc_coordinates(time, windspeed)
        pwv = np.array([pwv_matrix[positions[0]], pwv_matrix[positions[1]], pwv_matrix[positions[2]], pwv_matrix[positions[3]], pwv_matrix[positions[4]]])
        return pwv

    def calc_coordinates(self, time, windspeed):
        """Calculates the positions at which the pwv needs to be taken. Five
        different positions are taken, to make sky chopping and nodding possible
        in 2 directons.

        Returns
        ------------
        Positions: array
            The positions (gridpoints) at which the pwv needs to be taken.
        """
        grid_dif = int(round(self.separation/self.grid)) #number of gridpoints difference between positions
        distance = time*windspeed
        x_index = (int(round(distance/self.grid)))
        y_index = 14 #15th value, 3 m above the bottom of the map
        pos_1 = x_index, y_index
        pos_2 = (x_index + grid_dif), y_index
        pos_3 = (x_index + 2*grid_dif), y_index
        pos_4 = (x_index + grid_dif), y_index + grid_dif
        pos_5 = (x_index + grid_dif), y_index - grid_dif
        positions = [pos_1, pos_2, pos_3, pos_4, pos_5]
        return positions

##------------------------------------------------------------------------------
## The methods below are not used in the model atm
##------------------------------------------------------------------------------

    def make_image(self):
        fig = plt.figure()
        ax = plt.subplot(111)
        im = plt.imshow(self.pwv_matrix)
        # plt.gcf().subplots_adjust(top=0.83)
        ax.set_xlabel("[m]")
        ax.set_ylabel("[m]")
        ax_new = ax.twinx().twiny()
        ax_new.set_ylabel("arcsec")
        ticks = np.linspace(0, 1, 5)
        ticklabels = np.round(use_ARIS.m2arcsec(ticks*512)) #hardcoded
        plt.xticks(ticks, ticklabels)
        plt.yticks(ticks, ticklabels)
        ax_new.set_xlabel("arcsec")

        # secax = ax.secondary_yaxis('right', functions=(use_ARIS.m2arcsec, use_ARIS.arcsec2m))
        ax_new.set_title("Atmosphere Structure", fontsize=20)
        divider = make_axes_locatable(ax_new)
        # cax = divider.append_axes("right", size="4%", pad=0)
        # on the figure total in precent [left, bottom, width, height]
        cax = fig.add_axes([0.95, 0.1, 0.02, 0.75])
        colorbar_1 = fig.colorbar(im, cax=cax)
        colorbar_1.set_label('pwv in mm', labelpad=-10, y=1.05, rotation=0)
        plt.show()

    def m2arcsec(x):
        arcsec = np.arctan(x/1e3)*3600
        return arcsec

    def arcsec2m(x):
        m = np.tan(x/3600)*1e3
        return m

    def make_animation(self, length_side):
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xlabel("[m]")
        ax.set_ylabel("[m]")
        ax.set_title("Atmosphere Structure")
        pwv_shape = self.pwv_matrix.shape
        num_frames = min(pwv_shape[0], pwv_shape[1])-length_side
        y_min = int(np.round(pwv_shape[0]/2) - np.round(length_side/2))
        y_max = int(np.round(pwv_shape[0]/2) + np.round(length_side/2))
        x_min = 0
        x_max = length_side
        self.pwv_matrix = self.pwv_matrix[y_min:y_max, :]
        # pwv_frame = self.pwv_matrix[x_min:x_max, y_min:y_max]
        # im = plt.imshow(pwv_frame, cmap = 'viridis')
        ims = []
        for i in range(0, num_frames):
            # pwv_frame = self.pwv_matrix[y_min:y_max, x_min:x_max]
            pwv_frame = self.pwv_matrix[:, x_min:x_max]
            im = plt.imshow(pwv_frame, animated=True, cmap = 'viridis', vmin = np.min(self.pwv_matrix), vmax = np.max(self.pwv_matrix))
            ims.append([im])
            x_min += 1
            x_max += 1
        #
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar_1 = plt.colorbar(im, cax=cax)
        colorbar_1.set_label('pwv in mm', labelpad=-10, y=1.05, rotation=0)
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
        Writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1000)
        file_string = r'C:/Users/Esmee/Documents/BEP/DESHIMA/Animations/animation_atmosphere.mp4'
        ani.save(file_string, writer=Writer)
        plt.show()
        plt.pause(0.05)

# filename = "sample00.dat"
# length_side = 100
# atm_data_1 = use_ARIS(filename)
# # atm_data_1.make_image()
# atm_data_1.make_animation(length_side)

#square, in the middle
#512 = 16 * 32
# pwv_shape = pwv_samp.shape
# y_min = int(np.round(pwv_shape[0]/2) - np.round(length_side/2))
# y_max = int(np.round(pwv_shape[0]/2) + np.round(length_side/2))
# x_min = 0
# x_max = length_side
# pwv_samp_frame1 = pwv_samp[x_min:x_max, y_min:y_max]
# print(pwv_samp_frame1)
#use submatrix or splice or something, then use animate or time/update
# ax = plt.subplot(111)
# im = ax.imshow(pwv_samp, cmap = 'viridis')
#
# ax.set_xlabel("[m]")
# ax.set_ylabel("[m]")
# ax.set_title("Atmosphere Structure (pwv)")
#
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# plt.show()
# plt.savefig("./figs/Compare/Atm_structure.png")
