import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = 'C:/FFmpeg/bin/ffmpeg.exe'
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import numpy as np
import matplotlib.animation as animation
from matplotlib.transforms import Transform
from matplotlib.ticker import (
    AutoLocator, AutoMinorLocator)

import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from Atmosphere_model_Kah_Wuy.PhaseScreen_1.aris import load_aris_output

class use_ARIS(object):
    def __init__(self, filename):
        self.filename = filename
        if __name__ == "__main__":
            self.pathname = "../Data/output_ARIS/"
        else:
            # cwd = os.getcwd()
            self.pathname ="C:/Users/sup-ehuijten/Documents/DESHIMA-model_18_12_19/Python/BEP/Data/output_ARIS/"
            # self.pathname = ""
        # self.dEPL_matrix, self.pwv_matrix = load_aris_output(self.pathname + self.filename)
        self.dEPL_matrix = load_aris_output(self.pathname + self.filename)[0]
        self.a = 6.3003663 #m
        self.pwv_0 = 1.0 #mm
        self.pwv_matrix = self.pwv_0 + (1/self.a * self.dEPL_matrix*1e-6)*1e3 #in mm
        self.pwv_matrix = np.array(self.pwv_matrix)
        self.pwv_shape = self.pwv_matrix.shape

    def obt_pwv(pwv_matrix, time, count, f_chop, windspeed):
        # separation = 2*0.5663 #m (116.8 arcseconds)
        # sampling_rate = 160
        # y = sampling_rate/f_chop #number of samples per part of sky
        # z = 2*y
        # chop = (count % z < y)
        # print(chop)
        grid = 1 #number of m for 1 gridpoint, as given to ARIS
        # grid_dif = round(separation/grid) #number of gridpoints difference between 'on' and 'off'
        distance = time*windspeed
        length_x = pwv_matrix.shape[0]
        # x_index = (int(round(distance/grid)) + chop * grid_dif) % length_x
        x_index = (int(round(distance/grid))) % length_x
        y_index = 250
        pwv = pwv_matrix[x_index, y_index]
        return pwv

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
        num_frames = min(self.pwv_shape[0], self.pwv_shape[1])-length_side
        y_min = int(np.round(self.pwv_shape[0]/2) - np.round(length_side/2))
        y_max = int(np.round(self.pwv_shape[0]/2) + np.round(length_side/2))
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
