import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

class AtmosphereAnimation(animation.TimedAnimation):
    def __init__(self, pwv_matrix, length_side):
        # self.bar = Bar('Progress', max=len(F_vector))
        self.pwv_matrix = pwv_matrix
        self.pwv_shape = self.pwv_matrix.shape
        self.length_side = length_side

        fig = plt.figure()
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel("[m]")
        self.ax.set_ylabel("[m]")
        self.ax.set_title("Atmosphere Structure (pwv)")
        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes("right", size="5%", pad=0.05)

        self.y_min = int(np.round(self.pwv_shape[0]/2) - np.round(self.length_side/2))
        self.y_max = int(np.round(self.pwv_shape[0]/2) + np.round(self.length_side/2))
        self.x_min = 0
        self.x_max = self.length_side
        self.pwv_frame = self.pwv_matrix[self.x_min:self.x_max, self.y_min:self.y_max]

        self.t = np.linspace(0, 495, 495)
        # pink_colors = ['deeppink', 'darkviolet', 'rebeccapurple', 'slateblue', 'crimson']
        # self.ax1.set_xlabel('Frequency(GHz)')
        # self.ax1.set_ylabel('Atmospheric transmission')
        # self.line1a = Line2D([], [], color='darkblue')
        # self.ax1.add_line(self.line1a)

        self.anim = animation.TimedAnimation.__init__(self, fig, interval=10, blit=False)

    def _draw_frame(self, framedata):
        self.x_min += 1
        self.x_max =+ 1
        self.pwv_frame = self.pwv_matrix[self.x_min:self.x_max, self.y_min:self.y_max]

    def new_frame_seq(self):
        self.im = self.ax.imshow(self.pwv_frame, cmap = 'viridis')
        plt.colorbar(self.im, cax=self.cax)
        return iter(range(self.t.size))

    def _init_draw(self):
        self.im = self.ax.imshow(self.pwv_frame, cmap = 'viridis')
        plt.colorbar(self.im, cax=self.cax)
