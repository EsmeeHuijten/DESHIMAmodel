import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm #colormap
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.ndimage.filters import gaussian_filter

class telescope_transmission(object):

    def __init__(self, x_min = -5., x_max = 5., y_min = -5., y_max = 5., num_points = 100):
        self.x = np.linspace(x_min, x_max, num_points)
        self.y = np.linspace(y_min, y_max, num_points)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

    def filter_with_Gaussian(self, pwv_matrix, beam_radius = 5.):
        std = np.sqrt((beam_radius**2) / (2.0*np.log(10)))
        truncate = beam_radius/std
        output = gaussian_filter(pwv_matrix, std, mode='mirror', truncate=truncate)
        return output

    def calc_Gaussian(self, std_x, std_y, beam_radius):
        A = 1/(2*np.pi*std_x*std_y)
        self.transmission = A * np.exp(-(self.xx**2/(2*std_x**2) + self.yy**2/(2*std_y**2)))
        help_vec_circ = (np.sqrt(self.xx**2 + self.yy**2) < beam_radius)
        self.transmission = self.transmission * help_vec_circ
        return self.transmission

    def plot_Gaussian(self, std_x, std_y, beam_radius, colormap = cm.viridis):
        transmission = self.calc_Gaussian(std_x, std_y, beam_radius)

        # fig = go.Figure(data=[go.Surface(z=transmission)])
        # fig.show()

        # fig = go.Figure(data=[go.Mesh3d(x=self.xx,
        #            y=self.yy,
        #            z=transmission,
        #            opacity=0.5,
        #            )])
        # fig.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(self.xx, self.yy, transmission,
                      cmap=colormap,
                      linewidth=0,
                      antialiased=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('transmission');
        # plt.show()
        return self.transmission

# telescope_1 = telescope_transmission(-5., 5., -5., 5., 100)
# beam_radius = 5.
# std_x = np.sqrt((beam_radius**2) / (2.0*np.log(10)))
# std_y = std_x
# telescope_1.plot_Gaussian(std_x, std_y, 5., cm.plasma)
# plt.show()
