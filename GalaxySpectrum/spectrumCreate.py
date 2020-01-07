##-----------------------------------------
## Code written and edited by Tom Bakx
## tjlcbakx@gmail.com
##-----------------------------------------


##-----------------------------------------
## Header imports & colours
##-----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import glob
import GiveSL as gl

orange = '#ff9500'#(1,0.584,0)
blue =  '#007aff'  #(0,.478,1) blue
green = '#4cd964'
red = '#ff3b30'
grey = '#8e8e93'   #(142./255,142./255,147./255)


##-----------------------------------------
## Fit a bb-template for redshift and amplitude
##-----------------------------------------


def tomModel(v,Amp,z,T_cold,T_hot,Ratio,Beta):
    # Give the flux densities of the modeled SED at the requested frequencies v, at redshift z
    v_rest = (1+z)*v
    return Amp*((blackBody(v_rest,T_hot)*(v_rest**Beta)) + Ratio*(blackBody(v_rest,T_cold)*(v_rest**Beta)))

def blackBody(v,T):
    h = 6.626e-34
    c = 3.0e8
    k = 1.38e-23
    from numpy import exp
    return (2*h*(v*v*v))/(c*c*exp((h*v)/(k*T)) - 1)

def giveAmplitude(model,measurement,uncertainty,sumsquaresum=False):
    # This calculates the amplitude algebraically, to speed up computation time by
    # Decreasing the number of variables by the number of observations
    var1 = 0.
    var2 = 0.
    if sumsquaresum:
        for i in range(model.shape[0]):
            var1 += measurement[i]/uncertainty[i]
            var2 += model[i]/uncertainty[i]
    else:
        for i in range(model.shape[0]):
            var1 += measurement[i]*model[i]/(uncertainty[i]**2)
            var2 += (model[i]/uncertainty[i])**2
    return var1/var2

def giveLuminosity(I,I_err,lam,redshift,T_cold,T_hot,Ratio,Beta):
    # This method can use the FD (I), FD error(I_err) and redshift to determine the luminosity
    import numpy as np
    frequencies = np.linspace(3e11,3.75e13,(375-2))
    Lsun = 3.846e26
    intensities = np.zeros([len(frequencies)])
    model = tomModel((3.e8/lam),1,redshift,T_cold,T_hot,Ratio,Beta)
    amplitude = (1.e-26)*(frequencies[1]-frequencies[0])*giveAmplitude(model,I,I_err)*(4*np.pi*(Dlpen(redshift,giveAnswerInMeters=True)**2))/Lsun
    for i in range(len(frequencies)):
        intensities[i] = amplitude*tomModel(frequencies[i],1,redshift,T_cold,T_hot,Ratio,Beta)
    return intensities.sum()


T_cold     = 20.379
T_hot      = 43.7
Ratio       = 28.67
Beta        = 1.97

##-----------------------------------------
## Give a cosmological distance
##-----------------------------------------

def Dlpen(redshift, giveAnswerInMeters = False):
	from numpy import sqrt
	from astropy.cosmology import Planck15 as cosmo
	Mpc = 3.0857e22
	Dl = cosmo.luminosity_distance(redshift).value
	if giveAnswerInMeters:
		return Dl*Mpc
	else:
		return Dl


##-----------------------------------------
## Fitting a Gaussian
##-----------------------------------------

### Fitting Gaussians
def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))



##-----------------------------------------
##
##-----------------------------------------

def giveSpectrumInclSLs(luminosity,redshift,fLow=100,fHigh=3000,dfArray=0.01,linewidth=300):
	# Luminosity in units of log(L_fir [L_sol])
	# fLow, fHigh, dfArray in units of GHz
	# linewidth in units of km/s
	# Output: freqArray -> frequencies in units of GHz
	# Output: spectrum -> the spectrum in units of Jy
	import numpy as np
	# Generate frequency array
	freqArray = np.linspace(fLow,fHigh,int(1+(fHigh-fLow)/dfArray),endpoint=True)
	# Create spectrum according to Bakx+2018
	spectrum = tomModel(freqArray*(1.e9),1,redshift,T_cold,T_hot,Ratio,Beta)
	# Normalize the flux to the given far-IR luminosity
	normLum = giveLuminosity(np.array([spectrum[0],spectrum[0]]),np.array([1,1]),((3.e8)/((1.e9)*np.array([freqArray[0],freqArray[0]]))),redshift,T_cold,T_hot,Ratio,Beta)
	spectrum = (10.**luminosity)*spectrum/normLum
	# Add the spectrum lines
	B,names = gl.LFIRtoSL(luminosity,redshift,0,giveNames='Table')
	for i in range(len(B)):
		specLine = gaus(freqArray,B[i,1]*(600/linewidth),B[i,0],B[i,0]*linewidth/(3.e5))
		spectrum += specLine
	return freqArray,spectrum


B,names = gl.LFIRtoSL(12,2,0,giveNames='Table')
print(names)

##-----------------------------------------
## Save the plots
##-----------------------------------------

frequency,spectrum=giveSpectrumInclSLs(12,2)
plt.loglog(frequency,spectrum,color=blue,label='Redshift = 2')
frequency,spectrum=giveSpectrumInclSLs(12,4)
plt.loglog(frequency,spectrum,color=grey,label='Redshift = 4')
frequency,spectrum=giveSpectrumInclSLs(12,6)
plt.loglog(frequency,spectrum,color=orange,label='Redshift = 6')
plt.legend(loc='best',facecolor='white',edgecolor='none')
plt.xlim([200,500])
plt.ylim([2e-4,2e-2])

##-----------------------------------------
## Save the plots
##-----------------------------------------

plt.xlabel('Frequency [GHz]')
plt.ylabel('Flux density [Jy]')
plt.tight_layout()
# plt.savefig('Spectrum.png')
# plt.savefig('Spectrum.pdf')
plt.show()
#plt.close()
