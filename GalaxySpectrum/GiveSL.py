### LFIRtoInt,
# Turn a LFIR [log Lo] and redshift into a flux density at observed wavelength on earth [m]
# Uses the MP file with SEDs

### InttoLFIR
# Turn a flux density [Jy] and redshift into the LFIR [log Lo] at observed wavelength on earth [m]
# Uses the MP file with SEDs

### LFIRtoSL
# Provide a list of frequencies [GHz], flux densities [Jy], and variances on the spectral lines
# Row 0 to 12: CO (1-0) to CO(13-12)
# Row 13 to 17: SIII, SiII, OIII, NIII, OI,
# Row 18 to 24: OIII, NII, OI, CII, CI, CI, NII
# Uses the Distance Calculator from the MP file

### tauCalculator,
# Returns frequency [GHz] and tau as a function of the Percipable Water Vapours [mm], and takes the closest value to
# between the values 0.3, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5. Outside of these extremes, we take the values 0.5 and 5, respectively.
# Needs the file pgplot.csv

### calculateMLDF
# Calculates the MDLF [Jy] for a given frequency [GHz] and PWV [mm]
# Needs pgplot.csv

### MicrometertoGHz,
# Turns the wavelength [um] to frequency [GHz]

### GHztoMicrometer,
# Turns the frequency [GHz] to wavelength [um]
import sys
sys.path.append('./GalaxySpectrum/')
def LFIRtoInt(LFIR,z,lam,sfr):
    # Turn a LFIR and redshift in an intensity at the frequency observed at earth
    import MP
    import numpy as np
    Mpc = 3.0857e22
    # Provide the correct column to read out
    col = int(np.floor((LFIR-8.9)*10))
    if col > 45:
        col = 45
    elif col<1:
        col = 1
    adj = 0.1*(((LFIR-8.9)*10)-col)
    # Look up the row
    if sfr == 1:
        wavelength, data = MP.impSEDhot()
    else:
        wavelength, data = MP.impSEDcold()
    list_wavelength = np.argsort(abs(wavelength - (lam/(1+z))))[0]
    # And now directly to calculating the intensity
    # print adj
    # print data[list_wavelength][col]
    # print z
    # print MP.Dlpen(z)
    intensity = (10**adj)*data[list_wavelength][col]*(10**26)*(1+z)/(4*np.pi*Mpc*Mpc*(MP.Dlpen(z)**2))
    return intensity


def InttoLFIR(Intensity,z,lam,sfr):
    '''
    Transform an intensity into an expected LFIR. Beware of K-correction and therefore possible
    degeneracies within the regime of ~ 450 to 1000 um.
    '''
    from numpy import pi, argsort, linspace, log10,zeros
    import MP
    Lum = Intensity * 4 * pi * (((3.0857e22)*MP.Dlpen(z))**2)*(10**(-26))/(1+z) # 1+z is the frequency spacing
    if sfr ==1:
        wavelength, data = MP.impSEDhot()
    else:
        wavelength, data = MP.impSEDcold()
    # Look at the correct wavelength row
    wave_pos = argsort(abs(wavelength - (lam/(1+z))))[0]
    # and look up the correct luminosity in that row
    int_pos = argsort(abs(data[wave_pos,:] - Lum))[0]
    # Corrections, because the value is probably in between
    listlum = linspace(9.,13.5,46)
    LFIR = listlum[int_pos-1] + log10(Lum/data[wave_pos,int_pos])
    return LFIR



def LFIRtoSL(Luminosity,z,variance,COlines='Kamenetzky',lines='Bonato',giveNames='No'):
    '''
    Input: Luminosity [log (Lo)], redshift (z), variance (0 - no OR 1 - yes), COlines (Kamenetzky OR Rosenboom), lines (Bonato OR Spinoglio), giveNames ('Table', 'Tex' OR 'No').
    This will produce a list of frequencies [GHz] and flux densities [Jy], with or without a random variance for the SLs:
    Row 0 to 12: CO (1-0) to CO(13-12)
    Row 13 to 17: SIII, SiII, OIII, NIII, OI,
    Row 18 to 24: OIII, NII, OI, CII, CI, CI, NII
    giveNames toggles returning the names of the spectral lines in a list of strings.
    '''
    import numpy as np
    from MP import Dlpen
    from scipy.stats import norm
    Dl = Dlpen(z,giveAnswerInMeters=True)
    Dlmpc = Dlpen(z,giveAnswerInMeters=False)
    c = 3.0e8
    velocity = 600. #km/s
    df = velocity/300000. #km/s
    lSun = 3.826e26
    outputArray = np.zeros([25,4])
    # Load spectral line intensities
    if COlines == 'Kamenetzky':
        if __name__ == "__main__":
            path = ''
        else:
            path = './GalaxySpectrum'
        slco = np.genfromtxt(path + './K17_Table7', skip_header=1, dtype=np.float, delimiter=", ", unpack = False)
    elif COlines == 'Rosenboom':
        slco = np.loadtxt(path + './COcoeff', dtype=np.float, delimiter=" ", unpack = False)
    else:
        print('Did not recognise the CO-lines library, will be using Kamenetzky')
        slco = np.genfromtxt(path + './K17_Table7', skip_header=1, dtype=np.float, delimiter=", ", unpack = False)
    if lines == 'Bonato':
        sl = np.loadtxt(path + './coeffBonato', dtype=np.float, delimiter="    ", unpack = False)
    elif lines == 'Spinoglio':
        sl = np.loadtxt(path + './coeff_spinoglio', dtype=np.float, delimiter=", ", unpack = False)
    else:
        print('Did not recognise the line-library, will be using Bonatos line estimates')
        sl = np.loadtxt(path + './coeffBonato', dtype=np.float, delimiter="    ", unpack = False)
    for i in range(13):
        outputArray[i,0] = ((1.e-9)*(i+1)*(115*(10**9))/(1+z)) # The CO lines 1-0 to 13-12
    outputArray[13,0] = ((1.e-9)*c/((1+z)*33.48e-6)) # SIII
    outputArray[14,0] = ((1.e-9)*c/((1+z)*34.82e-6)) # SiII
    outputArray[15,0] = ((1.e-9)*c/((1+z)*51.81e-6)) # OIII
    outputArray[16,0] = ((1.e-9)*c/((1+z)*57.32e-6)) # NIII
    outputArray[17,0] = ((1.e-9)*c/((1+z)*63.18e-6)) # OI
    outputArray[18,0] = ((1.e-9)*c/((1+z)*88.36e-6)) # OIII
    outputArray[19,0] = ((1.e-9)*c/((1+z)*121.9e-6)) # NII
    outputArray[20,0] = ((1.e-9)*c/((1+z)*145.5e-6)) # OI
    outputArray[21,0] = ((1.e-9)*c/((1+z)*157.7e-6)) # CII
    outputArray[22,0] = ((1.e-9)*c/((1+z)*370.5e-6)) # CI
    outputArray[23,0] = ((1.e-9)*c/((1+z)*609.6e-6)) # CI
    outputArray[24,0] = ((1.e-9)*c/((1+z)*205e-6))    # NII
    if variance != 0:
        # Create three random numbers per galaxy, used to create the log-normal distribution
        randvar = (np.random.random(3))
        # This creates a log-normal distribution, with a maximum deviation of 2 sigma
        randvar = norm.ppf(randvar*0.96 + 0.02)/norm.ppf(0.02)
        for i in range(13):
            outputArray[i,2] = (randvar[0])
        outputArray[13,2] = (randvar[2])
        outputArray[14,2] = (randvar[2])
        outputArray[15,2] = (randvar[0])
        outputArray[16,2] = (randvar[0])
        outputArray[17,2] = (randvar[1])
        outputArray[18,2] = (randvar[0])
        outputArray[19,2] = (randvar[0])
        outputArray[20,2] = (randvar[1])
        outputArray[21,2] = (randvar[1])
        outputArray[22,2] = (randvar[1])
        outputArray[23,2] = (randvar[1])
        outputArray[24,2] = (randvar[0])
        # Create three random numbers per galaxy, used to create the log-normal distribution
        randvar = (np.random.random(3))
        # This creates a log-normal distribution, with a maximum deviation of 2 sigma
        randvar = norm.ppf(randvar*0.96 + 0.02)/norm.ppf(0.02)
        for i in range(13):
            outputArray[i,3] = (randvar[0])
        outputArray[13,3] = (randvar[2])
        outputArray[14,3] = (randvar[2])
        outputArray[15,3] = (randvar[0])
        outputArray[16,3] = (randvar[0])
        outputArray[17,3] = (randvar[1])
        outputArray[18,3] = (randvar[0])
        outputArray[19,3] = (randvar[0])
        outputArray[20,3] = (randvar[1])
        outputArray[21,3] = (randvar[1])
        outputArray[22,3] = (randvar[1])
        outputArray[23,3] = (randvar[1])
        outputArray[24,3] = (randvar[0])
    if lines == 'Spinoglio':
        # This part calculates the lines according to the Spinoglio
        # Two parts of the variance are taken into account, one on the steepness of the L_FIR - L_SL relation (outputArray[i,3])
        # and one on the total amplitude of the L_FIR - L_SL relation
        for i in range(13,22):
            #outputArray = (sigma ^ variance) * (Lsun * (10^ L_Spin_var) / freq-width) * (10^26 [Jy] / 4 pi Dl^2)
            outputArray[i,1]= ((10**sl[i-13,3])**outputArray[i,2]) * (lSun*(10**((sl[i-13,0] + outputArray[i,3]*sl[i-13,1])* Luminosity - sl[i-13,2]))/(df*outputArray[i,0]*1.e9)) * ((10**26)/(Dl*Dl*4.*np.pi))
    else:
        # This part calculates the lines according to Bonato
        # This method only has the variance on the total amplitude, as the steepness is fixed to 1
        for i in range(13,25):
            #outputArray = variance * (lum / f-width) * (10^26 / 4 pi Dl^2)
            outputArray[i,1]= (sl[i-13,2]**outputArray[i,2]) * (lSun*(10**(sl[i-13,0] * Luminosity - sl[i-13,1]))/(df*outputArray[i,0]*1.e9)) * ((10**26)/(Dl*Dl*4.*np.pi))
    if COlines != 'Rosenboom':
        # This method relies on many galaxies, documented in Kamenetzky
        # They use the unwieldy units of Laccent, which with a bit of effort can give the S_CO
        # The amplitude and the steepness are fitted in the paper, and simulated here.
        for i in range(13):
            # outputArray = StaceyValue * mult.factor * df_CII / df_CO * flux_density_CII
            Laccent = (Luminosity - slco[i,2] - outputArray[i,3]*slco[i,3])/(slco[i,0] + outputArray[i,2]*slco[i,1])
            outputArray[i,1] = (10**Laccent) *((1+z)*((115.*(i+1))**2))/((3.25e7)*(velocity)*(Dlmpc*Dlmpc))
    else:
        # From Rosenboom's paper, we extract the ratios in luminosity of each line, and then relate that to the CII luminosity
        # The variation is exactly that of the CII-line. The CII / CO (1-0) relation is from Stacey.
        # There is no scatter inside the CO-ladder
        for i in range(13):
            outputArray[i,1] = (1./4100.)*slco[i] * (outputArray[21,0] / outputArray[i,0]) * outputArray[21,1]
    if lines == 'Spinoglio':
        i = 24
        # NII_205 -> I need more data on how to calculate this property
        outputArray[24,1] = (2**outputArray[i,2]) * (lSun*(10**(1.05 * Luminosity - 4.747))/(df*outputArray[i,0]*1.e9)) * ((10**26)/(Dl*Dl*4.*np.pi))
        #
        outputArray[22,1] = 0.4 * outputArray[3,1]
        outputArray[23,1] = 0.3 * outputArray[6,1]
    if giveNames == 'Table':
        listOfSLs = ['CO (1-0)','CO (2-1)','CO (3-2)','CO (4-3)','CO (5-4)','CO (6-5)','CO (7-6)','CO (8-7)','CO (9 - 8)','CO (10 - 9)','CO (11-10)','CO (12 - 11)','CO (13-12)', 'SIII 33', 'SiII 35', 'OIII 52', 'NIII 57', 'OI 63', 'OIII 88', 'NII 122', 'OI 145', 'CII 158', 'CI 370', 'CI 610', 'NII 205']
        return outputArray,listOfSLs
    elif giveNames =='Tex':
        listOfSLs = 'CO (1 - 0) & CO (2 - 1) & CO (3 - 2) & CO (4 - 3) & CO (5 - 4) & CO (6 - 5) & CO (7 - 6) & CO (8 - 7) & CO (9 - 8) & CO (10 - 9) & CO (11 - 10) & CO (12 - 11) & CO (13 - 12) & SIII 33 & SiII 35 & OIII 52 & NIII 57 & OI 63 & OIII 88 & NII 122 & OI 145 & CII 158 & CI 370 & CI 610 & NII 205'
        return outputArray,listOfSLs
    else:
        return outputArray



def GHztoMicrometer(GHz, shouldRoundResult = False):
    return round(3e5/GHz) if shouldRoundResult else 3e5/GHz

def MicrometertoGHz(mum, shouldRoundResult = False):
    return round(3e5/mum) if shouldRoundResult else 3e5/mum

def tauCalculator(PWV,printTau='False'):
    '''
    This program provides the Tau for different Percipable Water Vapours, and linearly interpolates
    between the values 0.5, 1, 2, 6. Outside of these extremes, we take the values 0.5 and 6, respectively.
    '''
    import numpy as np
    if 'trans' not in globals():
        global trans
        trans = np.loadtxt('./pgplot.csv', dtype=np.float, delimiter=",", unpack = False)
    PWVARRAY = np.array([0.3, 0.5, 0.75, 1., 1.5, 2., 2.5, 3., 4., 5.])
    a = np.argmin(abs(PWVARRAY - PWV))
    if printTau == True:
        print('PWV  = ' + str(PWVARRAY[a]))
    outputArray = np.zeros([8301,2])
    outputArray[:,1] = trans[:,1+a]
    outputArray[:,0] = trans[:,0]
    return outputArray

def calculateMDLF(Frequency, PWV,roughness=18,Disp=400.,t_hours=1.,printTau=False):
    # Frequency in GHz, PWV in mm, roughness in um
    # Return the MDLF in Jy
    # Load in dependencies
    from numpy import sqrt, pi,exp
    from scipy.interpolate import interp1d
    # Make sure we are within operating frequencies
    if Frequency < 70 or Frequency > 900:
        print('Index out of bounds')
        return 10000
    # Return the opacity at a certain PWV and frequency
    A = tauCalculator(PWV)
    function = interp1d(A[:,0],A[:,1])
    eta = function(Frequency)
    if eta < 0.001:
        eta = 0.001
        if printTau == True:
            print('Eta interpolated to 0.0001')
    # Constants needed for calculations
    kb = 1.38e-23
    c = 2.99792e8
    h = 6.626e-34
    qe = 1.60217662e-19
    # Get units to SI
    time = 3600.0 * t_hours
    f = Frequency*1.e9
    df = (f/Disp)
    # Telescope and instrument parameters
    diatel = 10.0
    Tamb = 269.0
    Tspill = 269.0
    Tcabin = 280.0
    nf = 0.96
    nfi = 0.8
    npol = 0.5
    ncryo = 0.5
    nlens = 0.9
    nIFB = 0.8
    nsam = 1.0
    ninst = npol*ncryo * nlens* nIFB * nsam
    sn = 5.0
    # Calculate the sky temperature
    Tsky = Tamb*(1-eta);
    # Calculate the telescope spillover
    TeffTel = nf*Tsky + (1-nf)*Tspill
    # Aperture efficiency with Ruze's formula
    nA = 0.6*exp(4*pi*roughness*(1e-6)*(f/c))
    # Calculate the finite spillover from cabin
    TeffCabin = nfi*(TeffTel) + (1-nfi)*Tcabin
    # Calculate effective total temperature
    Teff = TeffCabin * ninst
    # Calculate the Noise Equivalent Source Power (NESP)
    NESP = sqrt(2. * kb * df * Teff * (h * f * (1 + (kb * Teff / (h * f))) + 0.000188 * qe / 0.57)) / (eta * nA * nfi * ninst)
    # Calculate the MDLF
    MDLF = (10 ** 26) * NESP * sn / (df * pi * ((diatel / 2) ** 2) * sqrt(time))
    return MDLF


def calculateNEWMDLF(Frequency, PWV,roughness=42,Disp=400.,t_hours=1.,printTau=False):
    # Frequency in GHz, PWV in mm, roughness in um
    # Return the MDLF in Jy
    # Load in dependencies
    from numpy import sqrt, pi,exp
    from scipy.interpolate import interp1d
    # Make sure we are within operating frequencies
    if Frequency < 70 or Frequency > 900:
        print('Index out of bounds')
        return 10000
    # Return the opacity at a certain PWV and frequency
    A = tauCalculator(PWV)
    function = interp1d(A[:,0],A[:,1])
    eta = function(Frequency)
    if eta < 0.001:
        eta = 0.001
        if printTau == True:
            print('Eta interpolated to 0.0001')
    # Constants needed for calculations
    kb = 1.38e-23
    c = 2.99792e8
    h = 6.626e-34
    qe = 1.60217662e-19
    # Get units to SI
    time = 3600.0 * t_hours
    f = Frequency*1.e9
    df = (f/Disp)
    # Telescope and instrument parameters
    diatel = 10.0
    Tamb = 269.0
    Tspill = 280.0
    Tcabin = 280.0
    nf = 0.89#0.96
    nfback = 0.52#
    nfi = 0.98
    npol = 0.5
    ncryo = 0.4
    nlens = 0.8
    nIFB = 0.4
    nsam = 1.0
    ninst = npol*ncryo * nlens* nIFB * nsam
    sn = 5.0
    # Calculate the sky temperature
    Tsky = Tamb*(1-eta);
    # Calculate the telescope spillover
    TeffTel = nf*Tsky + (1-nfback)*Tspill
    # Aperture efficiency with Ruze's formula
    nA = 0.6*exp(4*pi*roughness*(1e-6)*(f/c))
    # Calculate the finite spillover from cabin
    TeffCabin = nfi*(TeffTel) + (1-nfi)*Tcabin
    # Calculate effective total temperature
    Teff = TeffCabin * ninst
    # Calculate the Noise Equivalent Source Power (NESP)
    NESP = sqrt(2. * kb * df * Teff * (h * f * (1 + (kb * Teff / (h * f))) + 0.000188 * qe / 0.57)) / (eta * nA * nfi * ninst)
    # Calculate the MDLF
    MDLF = (10 ** 26) * NESP * sn / (df * pi * ((diatel / 2) ** 2) * sqrt(time))
    return MDLF


def giveTime(i,z,L):
    A = LFIRtoSL(L,z,0)
    return (calculateNEWMDLF(A[i,0],1.0)/A[i,1])**2
