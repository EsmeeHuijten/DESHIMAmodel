def impSEDcold():
    import numpy as np
    if 'frqSEDcold' not in globals():
        global frqSEDcold
        global intSEDcold
        path = './plot_galaxyspectrum_template.cold.dat'
        data = np.loadtxt(path, dtype=np.float, delimiter="   ", unpack = False)
        frqSEDcold = data[:,0]
        intSEDcold = data[:,1::]
    return frqSEDcold, intSEDcold

def impSEDhot():
    import numpy as np
    if 'frqSEDhot' not in globals():
        global frqSEDhot
        global intSEDhot
        path = './plot_galaxyspectrum_template.dat'
        data = np.loadtxt(path, dtype=np.float, delimiter="   ", unpack = False)
        frqSEDhot = data[:,0]
        intSEDhot = data[:,1::]
    return frqSEDhot, intSEDhot

def returnRedshiftAndLuminosity(randomValue):
    from numpy.random import random
    import numpy as np
    from scipy.interpolate import interp1d
    from GiveSL import InttoLFIR
    if 'CDFMeanFile' not in globals():
        global CDFMeanFile, redshiftRange, fluxDensityRange
        pathCDF = './CDFmeanfile.txt'
        pathZ = './redshiftmean.txt'
        pathFD = './fluxdensitymean.txt'
        CDFMeanFile = np.loadtxt(pathCDF, dtype=np.float, delimiter=' ', unpack = False)
        redshiftRange = np.loadtxt(pathZ, dtype=np.float, delimiter=' ', unpack = False)
        fluxDensityRange = np.loadtxt(pathFD, dtype=np.float, delimiter=' ', unpack= False)
    index = np.unravel_index((np.abs(CDFMeanFile - randomValue)).argmin(),(120, 300))
    # Just to make sure the randomvalue is in between the extreme values:
    if (randomValue - CDFMeanFile[index[0],-1])*(randomValue - CDFMeanFile[index[0],0]) < 0:
        interpolationFunction = interp1d(CDFMeanFile[index[0], :], redshiftRange[:])
        redshift = interpolationFunction(randomValue) + 0.0
    else:
        redshift = redshiftRange[10]
        print('The value was outside of the interpolation area, just given z: ' + str(redshift))
    fluxDensity = fluxDensityRange[index[0]]
    luminosityCold = InttoLFIR(fluxDensity, redshift, 850e-6, 0)
    luminosityHot = InttoLFIR(fluxDensity, redshift, 850e-6, 1)
    SFR = hotOrCold(luminosityCold, luminosityHot, redshift, random())
    if SFR:
        luminosity = luminosityHot
    else:
        luminosity = luminosityCold
    return luminosity, redshift, SFR


def hotOrCold(Lcold, Lhot, redshift, randomValue):
    from math import exp, log10, tanh
    if redshift > 2:
        B1 = (3.234e-3)*0.240623*((1+redshift)**(-0.919))
        A1 = (2.377e10)*49.6733408*((1+redshift)**(0.145))
    elif redshift > 0.879:
        B1 = (3.234e-3)*83.7459*((1+redshift)**(-6.246))
        A1 = (2.377e10)*0.320024014*((1+redshift)**4.737)
    else:
        B1 = (3.234e-3)*((1+redshift)**0.774)
        A1 = (2.377e10)*((1+redshift)**2.931)
    C1 = B1*((((10**Lcold)/(A1*2.377e10)))**(1-1.223)) * exp((-1/(2*(0.406**2)))*(log10((1+(10**(Lcold - log10(A1*2.377e10)))))**2))
    D1 = B1*((((10**Lhot)/(A1*2.377e10)))**(1-1.223)) *exp((-1/(2*(0.406**2)))*(log10((1+(10**(Lhot - log10(A1*2.377e10)))))**2))
    E1 = C1*(1-tanh((Lcold - (log10(23.677e10)))/0.572))/2.
    F1 = D1*(1+tanh((Lhot - (log10(23.677e10)))/0.572))/2.
    if randomValue > F1/(E1+F1):
        return 0
    else:
        return 1

from astropy.cosmology import Planck15

def Dlpen(redshift, giveAnswerInMeters = False):
    from numpy import sqrt
    redshift += 0.0
    Mpc = 3.0857e22
    Dl = Planck15.luminosity_distance(redshift).value
    if giveAnswerInMeters:
        return Dl*Mpc
    else:
        return Dl

