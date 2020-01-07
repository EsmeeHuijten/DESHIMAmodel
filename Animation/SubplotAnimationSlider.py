import matplotlib.animation as animation
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

class SubplotAnimationSlider(animation.FuncAnimation):
    def __init__(self, F_vector, eta_atm, Pkid, Tb_sky):
        fig = plt.figure()
        self.ax1 = fig.add_subplot(2, 1, 1)
        self.ax2 = fig.add_subplot(2, 1, 2)

        self.F_vector = F_vector
        self.F_vector_string = []
        for el in self.F_vector:
            self.F_vector_string.append("%.1f" % el)
        self.eta_atm = eta_atm
        self.Pkid = Pkid
        self.Tb_sky = Tb_sky

        self.t = np.linspace(0, 350, 350)

        self.ax1.set_xlabel('Frequency in GHz')
        self.ax1.set_ylabel('$\eta$')
        self.line1a = Line2D([], [], color='slateblue')
        self.line1b = Line2D([], [], color='deeppink')
        self.line1c = Line2D([], [], color='darkviolet')
        self.line1d = Line2D([], [], color='rebeccapurple')
        self.line1e = Line2D([], [], color='crimson')
        self.ax1.add_line(self.line1a)
        self.ax1.add_line(self.line1b)
        self.ax1.add_line(self.line1c)
        self.ax1.add_line(self.line1d)
        self.ax1.add_line(self.line1e)
        self.ax1.set_xlim(self.F_vector[0], self.F_vector[-1])
        self.ax1.set_ylim(0, 1)
        self.ax1.set_title("Atmospheric transmission versus frequency")

        self.ax2.set_xlabel('$P_{kid}$ in W')
        self.ax2.set_ylabel('$T_{sky}$ in K')
        self.line2a = Line2D([], [], color='darkblue')
        self.line2b = Line2D([], [], color='slateblue')
        self.line2c = Line2D([], [], color='dodgerblue')
        self.line2d = Line2D([], [], color='skyblue')
        self.ax2.add_line(self.line2a)
        self.ax2.add_line(self.line2b)
        self.ax2.add_line(self.line2c)
        self.ax2.add_line(self.line2d)
        self.ax2.set_xlim(0, np.max(self.Pkid))
        self.ax2.set_ylim(0, np.max(self.Tb_sky))
        self.ax2.set_title("Sky temperature versus KID power for frequency: " +  \
        self.F_vector_string[0] + " GHz")
        plt.subplots_adjust(hspace = 0.5)

        self.axamp = plt.axes([0.1, .03, 0.7, 0.02])
        # Slider
        self.samp = Slider(self.axamp, 'Filter', 0, 350, valinit=0)
        self.samp.on_changed(self._draw_frame)
        # self.samp.on_changed(print('changed'))
        animation.FuncAnimation.__init__(self, fig, self._draw_frame, \
        interval=50, blit=True)

    def _draw_frame_fake(self, framedata):
        return 0
    def _draw_frame(self, framedata):
        i = int(round(self.samp.val))

        self.line1a.set_data(self.F_vector, self.eta_atm[0, :])
        self.line1b.set_data(self.F_vector, self.eta_atm[1, :])
        self.line1c.set_data(self.F_vector, self.eta_atm[2, :])
        self.line1d.set_data(self.F_vector, self.eta_atm[3, :])
        self.line1e.set_data(self.F_vector[i] * np.ones(50), np.linspace(0, 1, 50))
        self.ax1.set_title("Atmospheric transmission versus frequency")

        self.line2a.set_data(self.Pkid[0, i, :], self.Tb_sky[0, i, :])
        self.line2b.set_data(self.Pkid[1, i, :], self.Tb_sky[1, i, :])
        self.line2c.set_data(self.Pkid[2, i, :], self.Tb_sky[2, i, :])
        self.line2d.set_data(self.Pkid[3, i, :], self.Tb_sky[3, i, :])
        self.ax2.set_title("Sky temperature versus KID power for frequency: " +  \
        self.F_vector_string[i] + " GHz")

        return [self.line1a, self.line1b,self.line1c,
                self.line1e, self.line1d, self.line2a,
                self.line2b, self.line2c, self.line2d]
