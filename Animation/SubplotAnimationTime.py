import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import interpolate
from matplotlib.lines import Line2D
from progress.bar import Bar

class SubplotAnimationTime(animation.TimedAnimation):
    def __init__(self, F_vector, eta_atm, Pkid, Tb_sky, pwv_vector):
        self.bar = Bar('Progress', max=len(F_vector))
        self.pwv_vector = pwv_vector

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
        pink_colors = ['deeppink', 'darkviolet', 'rebeccapurple', 'slateblue', 'crimson']
        self.ax1.set_xlabel('Frequency(GHz)')
        self.ax1.set_ylabel('Atmospheric transmission')
        self.line1a = Line2D([], [], color='darkblue')
        self.line1b = Line2D([], [], color='slateblue')
        self.line1c = Line2D([], [], color='dodgerblue')
        self.line1d = Line2D([], [], color='skyblue')
        self.line1e = Line2D([], [], color='crimson')
        self.ax1.add_line(self.line1a)
        self.ax1.add_line(self.line1b)
        self.ax1.add_line(self.line1c)
        self.ax1.add_line(self.line1d)
        self.ax1.add_line(self.line1e)
        self.ax1.set_xlim(self.F_vector[0], self.F_vector[-1])
        self.ax1.set_ylim(0, 1)
        self.ax1.legend(['EL = 20.0', 'EL = 43.3', 'EL = 66.6', 'EL = 90.0'], loc='lower left')
        # self.ax1.set_title("Atmospheric transmission versus frequency")

        self.ax2.set_xlabel('KID power(pW)')
        self.ax2.set_ylabel('Sky temperature(K)')
        self.line2a = Line2D([], [], color='darkblue')
        self.line2b = Line2D([], [], color='slateblue')
        self.line2c = Line2D([], [], color='dodgerblue')
        self.line2d = Line2D([], [], color='skyblue')
        self.line2e = Line2D([], [], color = 'crimson', linewidth = 0, marker = ".", markersize = 0.5)
        self.ax2.add_line(self.line2a)
        self.ax2.add_line(self.line2b)
        self.ax2.add_line(self.line2c)
        self.ax2.add_line(self.line2d)
        self.ax2.add_line(self.line2e)
        self.ax2.set_xlim(0, np.max(self.Pkid)*1e12)
        self.ax2.set_ylim(0, np.max(self.Tb_sky))
        self.ax2.set_title("Frequency: " +  \
        self.F_vector_string[0] + " GHz")
        plt.subplots_adjust(hspace = 0.5)

        self.anim = animation.TimedAnimation.__init__(self, fig, interval=10, blit=False)

    def _draw_frame(self, framedata):
        i = framedata
        pwv_index = [0, 4, 9, 14]
        EL_values = np.linspace(20., 90., 25)
        EL_index = [0, 8, 16, 24]
        # obtain data interpolation
        f_load = np.load(r'C:\Users\Esmee\Documents\BEP\DESHIMA\Python\BEP\Data\splines_Tb_sky\spline_' \
        + '%.1f' % (self.F_vector[i]) +'GHz.npy')
        # print('f_load', f_load)
        f_function = f_load.item()
        temperature_tot = np.array([])
        power_tot = np.array([])
        errorbars = np.array([len(EL_index), 25])
        for i in range(0, len(EL_index)):
            el = EL_index[i]
            power = np.sort(self.Pkid[:, i, el])
            power_tot = np.append(power_tot, power)
            # if i in self.peak_indices:
            temperature = f_function(EL_values[el], power)
            errorbars[i, :] = abs(temperature-self.Tb_sky[:, i, EL_index[el]])
            # else:
            #     temperature = f_function(power)
            temperature_tot = np.append(temperature_tot, temperature)

        self.line1a.set_data(self.F_vector, self.eta_atm[pwv_index[0], :])
        self.line1b.set_data(self.F_vector, self.eta_atm[pwv_index[1], :])
        self.line1c.set_data(self.F_vector, self.eta_atm[pwv_index[2], :])
        self.line1d.set_data(self.F_vector, self.eta_atm[pwv_index[3], :])
        self.line1e.set_data(self.F_vector[i] * np.ones(50), np.linspace(0, 1, 50))

        # self.line2a.set_data(self.Pkid[pwv_index[0], i, :]*1e12, self.Tb_sky[pwv_index[0], i, :])
        # self.line2b.set_data(self.Pkid[pwv_index[1], i, :]*1e12, self.Tb_sky[pwv_index[1], i, :])
        # self.line2c.set_data(self.Pkid[pwv_index[2], i, :]*1e12, self.Tb_sky[pwv_index[2], i, :])
        # self.line2d.set_data(self.Pkid[pwv_index[3], i, :]*1e12, self.Tb_sky[pwv_index[3], i, :])
        self.line2a.set_data(self.Pkid[:, i, EL_index[0]]*1e12, self.Tb_sky[:, i, EL_index[0]], yerr = errorbars[0, :])
        self.line2b.set_data(self.Pkid[:, i, EL_index[1]]*1e12, self.Tb_sky[:, i, EL_index[1]])
        self.line2c.set_data(self.Pkid[:, i, EL_index[2]]*1e12, self.Tb_sky[:, i, EL_index[2]])
        self.line2d.set_data(self.Pkid[:, i, EL_index[3]]*1e12, self.Tb_sky[:, i, EL_index[3]])
        self.line2e.set_data(power_tot*1e12, temperature_tot)
        self.ax2.set_title("Frequency: " +  \
        self.F_vector_string[i] + " GHz")

        self._drawn_artists = [self.line1a, self.line1b,self.line1c,
                               self.line1e, self.line1d, self.line2a,
                               self.line2b, self.line2c, self.line2d, self.line2e] #, self.label]
        plt.savefig(r"C:\Users\Esmee\Documents\BEP\DESHIMA\Animations\animation_frames_new\frame" + str(i))
        self.bar.next()
        if framedata == len(self.F_vector):
            self.bar.finish()

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines = [self.line1a, self.line1b, self.line1c,
                 self.line1d, self.line1e, self.line2a,
                 self.line2b, self.line2c, self.line2d, self.line2e]
        for l in lines:
            l.set_data([], [])
