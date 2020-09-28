
from numpy import *
import matplotlib.pylab as plt

## subroutines
##............................................................

def co_luminosity(z, d_l, f_obs, int):
	
	c1, c2 = 3.25e+07, 1.04e-03
	
	#Lp  = c1 * int * d_l**2 / (f_obs**2 * (1 + z)**3 )
	L   = c2 * int * d_l**2 * f_obs
	
	return L


def flux_from_line_luminosity(z, d_l, f_obs, L):
	"""
		d_l - luminosity distance (Mpc)
		f_obs - observing frequency (GHz)
		L - line luminosity (L_Sun)
	"""
	
	L_for1Jykms = co_luminosity(z, d_l, f_obs,  1.000) ## Lsun
	F_for1Jykms = 1.000 * 1e-26 * (f_obs  * 1e9 / 299792.458) ## W m^-2

	return L * (F_for1Jykms / L_for1Jykms)


## user settings
##............................................................

## total infrared luminosity (L_Sun)
Ltir = 1.e+11

## would you like to use line-to-TIR ratios for dwarf galaxies?
switch_dwarf = True

## line-to-TIR luminosity ratio (L_Sun or Watt)
Rcii_B08, Roiii_B08, Roi_B08 = 1.3e-3, 8.0e-4, 1.0e-3 ## from Brauher+2008
Rcii_DGS, Roiii_DGS, Roi_DGS = 2.5e-3, 5.0e-3, 1.7e-3 ## from Cormier+2015

## rest frequency (GHz)
f_cii, f_oiii, f_oi = 1900.5369, 3393.00062, 4744.8

## redshift
execfile('Dl_at_z2.py')
#z = concatenate( (arange(0.1,0.5,0.1), arange(0.5,2,0.5), arange(2,6,1)), 1)
#z = concatenate( (arange(0.1,0.5,0.1), arange(0.5,2,0.5), arange(2,12,1)), 1)
#z = concatenate( (arange(0.1,1,0.1), arange(1,8,0.2)), 1)
z = array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20,])
d_l = Dl_at_z(z) ## luminosity distance (Mpc)

Fcii_B08  = flux_from_line_luminosity(z, d_l, f_cii/(1+z),  Ltir * Rcii_B08)
Foiii_B08 = flux_from_line_luminosity(z, d_l, f_oiii/(1+z), Ltir * Roiii_B08)
Foi_B08   = flux_from_line_luminosity(z, d_l, f_oi/(1+z),   Ltir * Roi_B08)

Fcii_DGS  = flux_from_line_luminosity(z, d_l, f_cii/(1+z),  Ltir * Rcii_DGS)
Foiii_DGS = flux_from_line_luminosity(z, d_l, f_oiii/(1+z), Ltir * Roiii_DGS)
Foi_DGS   = flux_from_line_luminosity(z, d_l, f_oi/(1+z),   Ltir * Roi_DGS)



## MDLF esitmates
##............................................................

#execfile('nefd3_antarctica.py')
#execfile('nefd3_atacama.py')
execfile('nefd3_alma.py')

## Survey parameters (common)

t_total=1. #(hr) **dummy** (doesn't affect MDLF)
Asurvey=1. #(sq-deg) **dummy** (doesn't affect MDLF)

t_int_forMDLF = 10000. #(sec) integration time used for MDLF

pwv_atacama = 600 #(micron)
Tam_atacama = 273 #(Kelvin)


## ALMA

sde=survey_alma(t_total=t_total, Asurvey=Asurvey,
				pwv=pwv_atacama, Tam=Tam_atacama,
				R=3000, Npix=1,
				Nant=50,
				)
sde.t_int = t_int_forMDLF
sde.get_nefd()


## plot
##............................................................

#plt.plot(sde.frq*1e-9, log10(sde.mdlf), color='0.60', linewidth=1)
plt.fill_between(sde.frq*1e-9, -40, log10(sde.rms * sde.df), color='0.40', alpha=0.5)
#plt.plot(sde.frq[1285:4175]*1e-9, log10(sde.mdlf[1285:4175]), color='lightseagreen', linewidth=2)

#print sde.frq[825]*1e-9
#print sde.frq[4175]*1e-9

## line

if switch_dwarf:
	Fcii_DGS, Foiii_DGS, Foi_DGS = Fcii_DGS, Foiii_DGS, Foi_DGS
else:
	Fcii_DGS, Foiii_DGS, Foi_DGS = Fcii_B08, Foiii_B08, Foi_B08

for i in range(len(z)):
	x = array( [f_cii, f_oiii] ) / (1 + z[i])
	y = array( [Fcii_DGS[i], Foiii_DGS[i]] )
	plt.plot(x, log10(y), '--', color='0.1')

for i in range(len(z)):
	x = array( [f_oi, f_oiii] ) / (1 + z[i])
	y = array( [Foi_DGS[i], Foiii_DGS[i]] )
	plt.plot(x, log10(y), '--', color='0.1')

plt.plot(f_cii/(1+z),  log10(Fcii_DGS),  "^-", linewidth=1.5, color='darkgoldenrod', markersize=9)
plt.plot(f_oiii/(1+z), log10(Foiii_DGS), "o-", linewidth=1.5, color='royalblue', markersize=9)
plt.plot(f_oi/(1+z),   log10(Foi_DGS),   "s-", linewidth=1.5, color='purple', markersize=9)

## cosmetics

#plt.legend(('LST MDLF','Tsukuba MDLF',), loc='upper left')
#plt.legend(('ASTE Sky','DESHIMA',), loc='upper left')

plt.xlim(0.05e+3,1.00e+3)
plt.ylim(-23.0,-18)

#plt.axvspan(240, 720, fc="seagreen", ec='none', alpha=0.3)
#plt.axvspan(325, 365, fc="blue", ec='none', alpha=0.3)



plt.xlabel("Frequency (GHz)", fontsize=14)
plt.ylabel("log flux (W m^-2)", fontsize=14)
#plt.xscale('log')
plt.rcParams["font.size"] = 12
plt.tight_layout()

#plt.show()

plt.savefig("plot.pdf")

#for i in range(len(sde.frq)):
#	print '%.1f %.3e' % (sde.frq[i]*1e-9, sde.rms[i]*1e26)

# EOF
