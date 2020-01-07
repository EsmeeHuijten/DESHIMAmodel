import shlex
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.insert(1, r'C:/Users/Esmee/Documents/BEP/DESHIMA/Python/BEP/DESHIMA/MKID')
import filterbank

def obt_data_from_am(filename, pwv_value, F_min, F_max, len_freq_vector):
    F_step = (440-220)/len_freq_vector
    # run am
    # usage: am Chajnantor.amc fmin[GHz] fmax[GHz] zenith_angle[deg] pwv[um] Tground[K]
    am_command = "am {0}.amc {1} {2} {3} {4}".format(filename, F_min, F_max, F_step, pwv_value)
    p = subprocess.Popen(shlex.split(am_command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_data, stderr_data = p.communicate()
    return p.returncode ,stdout_data, stderr_data

# result = obt_data_from_am("EsmeesFirstTryOutFile")
len_freq_vector = 100
F_min = 0
F_max = 15
filename = "EsmeesFirstTryOutFile"
pwv_vector = np.linspace(0, 3, 31)
# EPL_matrix = np.zeros([len(pwv_vector), len_freq_vector]) #vectors inside vectors
for i in range(0, len(pwv_vector)):
    ret, stdout, stderr = obt_data_from_am(filename, pwv_vector[i], F_min, F_max, len_freq_vector)
    freq, EPL = np.loadtxt(stdout.splitlines()).T
    if i == 0:
        EPL_matrix = np.zeros([len(pwv_vector), len(freq)])
    EPL_matrix[i, :] = EPL

# get eta_atm data
filterbank_1 = filterbank.filterbank(220e9, 500, 440e9, 100)
eta_atm_vector = filterbank_1.getPoints_etaF_curve(1.00, 90.0)

fig, ax1 = plt.subplots()
slopes = np.array([])
for k in range(0, len(freq)):
    slope_0 = stats.linregress(pwv_vector, EPL_matrix[:, k])[0]
    slopes = np.append(slopes, slope_0)
avg_slope = np.average(slopes)
print(avg_slope)
color = 'tab:red'
ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel('dEPL/dpwv', color=color)
ax1.plot(freq, slopes, label = 'slope', color=color)
ax1.plot(freq, EPL_matrix[10, :], '.', label='pwv = 1.00 mm', color='orange')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left', bbox_to_anchor=(0,0.85))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Atmospheric transmission', color=color)  # we already handled the x-label with ax1
ax2.plot(filterbank_1.F/1e9, eta_atm_vector, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# pwv_l = np.arange(100,3100,100)
# epl_350 = []
# tau_l = []
# Tsk_l = []
# index=2500
# for pwv in pwv_l:
#     ret, stdout, stderr = iam_datab(fmin=310, fmax=400, pwv=pwv)
#     freq_GHz, tau, Tb_A, EPL = np.loadtxt(stdout.splitlines()).T
#     Tsk_l.append(Tb_A)
#     epl_350.append(EPL[index])
#     tau_l.append(tau)
# Tsk_ar = np.array(Tsk_l)
# tau_sk_ar = np.array(tau_l)

# command:
#     am EsmeesFirstTryOutFile.amc pwv_value >EsmeesFirstTryOutFile.out
