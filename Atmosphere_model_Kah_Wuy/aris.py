from numpy import *
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import mlab


## 1. load aris output

def load_aris_output(infile, nx=512, ny=512):

	d=loadtxt(infile, delimiter=',')
	# print(len(d))
	epl=zeros((nx,ny))

	for i in range(len(d)):
		epl[int(d[i,0]),int(d[i,1])]=d[i,2]

	# Eq. 4, Asaki+05, ALMA memo #535
	# sigma_WVR [um] = 10 * (1 + PWV[mm])
	# => PWV = 0.1 * sigma_WVR - 1

	pwv = 0.1*epl - 1.

	return epl,pwv


## 2. convolution with near-field beam

def beamconvolve_pwv(pwv, nx=512, ny=512, sigma=2.35, plot=False):

	kernel = outer(signal.gaussian(nx, sigma), signal.gaussian(ny, sigma))
	pwv_conv=signal.fftconvolve(pwv, kernel, mode='same') / sum(kernel)

	if plot is True:
		plt.imshow(pwv_conv); plt.colorbar(); plt.show()

	return pwv_conv


## 3. convert 1-d PWV to intensity

def convert_pwv_to_intensity(pwv_conv, seed, fobs=350.e+9):

	nx,ny=pwv_conv.shape
	edge=6

	#I = pwv_to_I(pwv_conv, fobs, Tgr=273.)  # not defined yet
	I = pwv_conv[seed,edge:nx-edge-1] / mean(pwv_conv[seed,edge:ny-edge-1])  # just mimicing sky variation, but not correct...
	I = I * 1.61e-15  # radiance at 350.0 GHz, at PWV=1.0mm : 1.609627e-15 watt*m-2*Hz-1*sr-1

	return I

## 4. calculate loading power

def planck(f,T=273.):
	kB=1.3806504e-23
	h=6.62606896e-34
	c=299792458.
	return (2*h*f**3/c**2)/(exp(h*f/(kB*T))-1.)

# re_Aomega just use for calculate astronomical signal's Loading Power.
# You can just use Psky as Psig when re_Aomega is used.
def calculate_LP_signal(I, fobs=350.e+9, dfobs=1.e+9, plot=False, re_AOmega =0):

	kB=1.3806504e-23
	h=6.62606896e-34
	c=299792458.

	lobs = c/fobs
	AOmega = lobs**2
	"""
	if re_AOmega == 0:
		AOmega_sig = AOmega
	else :
		AOmega_sig = re_AOmega
	"""
	AOmega_sig = AOmega
	eta_bs,eta_cryo,eta_lensAnt = 0.99,0.5,0.09
	eta_inst = eta_cryo*eta_lensAnt
	eta_filterbank = 2.19

	Psky     = eta_bs*eta_inst           * I * dfobs * AOmega_sig
	Pspill   = (1.-eta_bs)*eta_inst      * planck(fobs,T=273.) * dfobs * AOmega
	Pcryo    = (1.-eta_cryo)*eta_lensAnt * planck(fobs,T=4.00) * dfobs * AOmega
	PlensAnt = (1.-eta_lensAnt)          * planck(fobs,T=0.12) * dfobs * AOmega

	Popt = Psky + Pspill + Pcryo + PlensAnt
	Popt = Popt / 2.  #accounting for single polarization

	T_eff = 30.  # eta_opt * Tsky ~ 30 K at 350 K
	B = (exp(h*fobs/kB/T_eff)-1.)**(-1)
	Delta_g, eta_pb = 0.1957*1.60218e-22, 0.4

	NEP_ph = sqrt( 2. * Popt * (h*fobs*(1.+B) + Delta_g/eta_pb) )

	if plot is True:
		plt.plot(NEP_ph, color='red'); plt.show()

	#return Psky,Pspill,Pcryo,PlensAnt,Popt,NEP_ph
	return Psky/2.


def calculate_loading_power(I, fobs=350.e+9, dfobs=1.e+9, plot=False):

	kB=1.3806504e-23
	h=6.62606896e-34
	c=299792458.

	lobs = c/fobs
	AOmega = lobs**2

	eta_bs,eta_cryo,eta_lensAnt = 0.99,0.5,0.09
	eta_inst = eta_cryo*eta_lensAnt

	Psky     = eta_bs*eta_inst           * I * dfobs * AOmega
	Pspill   = (1.-eta_bs)*eta_inst      * planck(fobs,T=273.) * dfobs * AOmega
	Pcryo    = (1.-eta_cryo)*eta_lensAnt * planck(fobs,T=4.00) * dfobs * AOmega
	PlensAnt = (1.-eta_lensAnt)          * planck(fobs,T=0.12) * dfobs * AOmega

	Popt = Psky + Pspill + Pcryo + PlensAnt
	Popt = Popt / 2.  #accounting for single polarization

	T_eff = 30.  # eta_opt * Tsky ~ 30 K at 350 K
	B = (exp(h*fobs/kB/T_eff)-1.)**(-1)
	Delta_g, eta_pb = 0.1957*1.60218e-22, 0.4

	NEP_ph = sqrt( 2. * Popt * (h*fobs*(1.+B) + Delta_g/eta_pb) )

	if plot is True:
		plt.plot(NEP_ph, color='red'); plt.show()

	return Psky,Pspill,Pcryo,PlensAnt,Popt,NEP_ph


## - main

if __name__ == '__main__':

	epl,pwv = load_aris_output("sample00.dat", nx=512, ny=512)
	pwv_conv = beamconvolve_pwv(pwv, nx=512, ny=512, sigma=2.35, plot=False)

	v_wind,grid_spacing=20.,1. #[m/s],[m]
	Fs=v_wind/grid_spacing
	PSD_final=zeros(PSD.shape)

	for i in arange(24):
		I = convert_pwv_to_intensity(pwv, 20*(i+1), fobs=350.e+9)
		Psky,Pspill,Pcryo,PlensAnt,Popt,NEP_ph = calculate_loading_power(I, fobs=350.e+9, dfobs=1.e+9)
		PSD,f = mlab.psd(Popt, NFFT=256, Fs=Fs, scale_by_freq=True)
		PSD_final += PSD
		plt.plot(log10(f),0.5*log10(PSD),'-',color='black')
		plt.plot(array([-1.,1.]),log10(array([NEP_ph,NEP_ph])),':',color='gray')

	plt.plot(log10(f),0.5*log10(PSD_final/24.),'-',color='red')
	plt.xlabel('log frequency (Hz)')
	plt.ylabel('log PSD (W Hz$^{-1/2}$)')
	plt.grid(True)
	plt.show()
