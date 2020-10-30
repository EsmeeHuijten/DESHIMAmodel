import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm #colormap
#from mpl_toolkits.mplot3d import Axes3D
#import plotly.graph_objects as go
from scipy.ndimage.filters import gaussian_filter

class telescope_transmission(object):
    """
    This class is used to calculate the Gaussian with which the beam of the
    ASTE 10 m Telescope is approximated. It can calculate and plot a Gaussian
    (used solely for imaging purposes, not for calculations in the model) and it
    can filter the atmosphere map obtained in use_ARIS with the Gaussian beam.
    """

    def __init__(self, x_min = -5., x_max = 5., y_min = -5., y_max = 5., num_points = 100):
        self.x = np.linspace(x_min, x_max, num_points)
        self.y = np.linspace(y_min, y_max, num_points)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

    def filter_with_Gaussian(self, pwv_matrix, gridsize, beam_radius = 5.):
        """filteres the pwv_matrix from use_ARIS with a Gaussian beam.

        Returns
        ------------
        filtered_pwv_matrix: array
            pwv matrix that has been filtered with a Gaussian
            Unit: m
        """
        beam_radius = beam_radius/gridsize
        std = np.sqrt((beam_radius**2) / (2.0*np.log(10)))
        truncate = beam_radius/std
        pwv_matrix = pwv_matrix
        filtered_pwv_matrix = gaussian_filter(pwv_matrix, std, mode='mirror', truncate=truncate)
        return filtered_pwv_matrix

    def calc_Gaussian(self, std_x, std_y, beam_radius):
        """Calculates the values of a Gaussian when the standard deviations and
        the beam radius are given. The Gaussian is made so that it is truncated
        at the beam radius of the telescope, and that the truncated edge has a
        height that is 10 per cent of the maximum height of the Gaussian (height
        at (0, 0)).

        Returns
        ------------
        self.transmission: array
            value of the Gaussian
        """
        A = 1/(2*np.pi*std_x*std_y)
        self.transmission = A * np.exp(-(self.xx**2/(2*std_x**2) + self.yy**2/(2*std_y**2)))
        help_vec_circ = (np.sqrt(self.xx**2 + self.yy**2) < beam_radius)
        self.transmission = self.transmission * help_vec_circ
        return self.transmission

    def plot_Gaussian(self, std_x, std_y, beam_radius, colormap = cm.viridis):
        """Plots the values of a Gaussian when the standard deviations and
        the beam radius are given. The Gaussian is made so that it is truncated
        at the beam radius of the telescope, and that the truncated edge has a
        height that is 10 per cent of the maximum height of the Gaussian (height
        at (0, 0)).

        Returns
        ------------
        self.transmission: array
            value of the Gaussian
        """
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

##------------------------------------------------------------------------------
## Code that might be useful later
##------------------------------------------------------------------------------

# telescope_1 = telescope_transmission(-5., 5., -5., 5., 100)
# beam_radius = 5.
# std_x = np.sqrt((beam_radius**2) / (2.0*np.log(10)))
# std_y = std_x
# telescope_1.plot_Gaussian(std_x, std_y, 5., cm.plasma)
# plt.show()
