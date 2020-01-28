import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = 'C:/FFmpeg/bin/ffmpeg.exe'
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

    a = 6.3003663 #m
    pwv_0 = 1.0 #mm
    separation = 2*0.5663 #m (116.8 arcseconds)
    x_length_strip = 32768

    def __init__(self, prefix_filename, grid, windspeed, time, loadCompletePwvMap = 0):
        self.prefix_filename = prefix_filename
        self.grid = grid
        self.windspeed = windspeed
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

    def load_complete_pwv_map(self):
        path = self.path_model + '/Data/output_ARIS/complete_filtered_pwv_map.txt'
        self.filtered_pwv_matrix = np.loadtxt(path)

    def initialize_pwv_matrix(self, time):
        max_distance = (time*self.windspeed + 2*self.separation)
        max_x_index = math.ceil(max_distance/self.grid)
        num_strips = min(math.ceil(max_x_index/self.x_length_strip), 40)
        print('num_strips', num_strips)
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
                # print('shape dEPL matrix part', self.dEPL_matrix.shape)
                # print('shape new part', epl.shape)
                self.dEPL_matrix = np.concatenate((self.dEPL_matrix, epl[:, 0:30]), axis = 0)
        print('shape EPL matrix', self.dEPL_matrix.shape)
        self.pwv_matrix = self.pwv_0 + (1/self.a * self.dEPL_matrix*1e-6)*1e3 #in mm
        self.pwv_matrix = np.array(self.pwv_matrix)

    def obt_pwv(self, time, count, windspeed):
        pwv_matrix = self.filtered_pwv_matrix
        length_x = pwv_matrix.shape[0]
        positions = self.calc_coordinates(time, windspeed)
        pwv = np.array([pwv_matrix[positions[0]], pwv_matrix[positions[1]], pwv_matrix[positions[2]], pwv_matrix[positions[3]], pwv_matrix[positions[4]]])
        return pwv

    def calc_coordinates(self, time, windspeed):
        grid_dif = int(round(self.separation/self.grid)) #number of gridpoints difference between positions
        distance = time*windspeed
        x_index = (int(round(distance/self.grid)))
        y_index = 14 #15th value, 3 m above the bottom of the map
        pos_1 = x_index, y_index
        pos_2 = (x_index + grid_dif), y_index
        pos_3 = (x_index + 2*grid_dif), y_index
        pos_4 = (x_index + grid_dif), y_index + grid_dif
        pos_5 = (x_index + grid_dif), y_index - grid_dif
        return [pos_1, pos_2, pos_3, pos_4, pos_5]

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
