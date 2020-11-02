"""
This module allows users to execute funtioncs in signal_transmitter, while providing default values so not all values need to be set manually.
"""


import numpy as np
import signal_transmitter as st
import DESHIMA.MKID.filterbank as ft
from pathlib import Path

#from . import signal_transmitter as st
#from .DESHIMA.MKID import filterbank as ft
import os

def calcMaxObsTime(dictionary):
    """
    Calculates the maximum observing time, using the number of gridpoints in
    one atmosphere strip, the gridwidth, the number of atmosphere strips and the
    windspeed
    """
    # maximum time
    # every strip has 32768 x-values
    max_obs_time = (dictionary['x_length_strip'] - 3 * dictionary['separation'])* \
                    dictionary['max_num_strips']*dictionary['grid'] \
                    /dictionary['windspeed']
    return max_obs_time

def convert_folder(folder):
    folder = folder.strip('/')
    folder = folder.strip('\\')
    sourcepath = Path.cwd()
    while folder.startswith('.'):
        folder = folder.strip('.')
        folder = folder.strip('/')
        folder = folder.strip('\\')
        sourcepath = sourcepath.parent
    sourcepath = sourcepath.joinpath(folder)
    return sourcepath

def new_filterbank(dictionary):
    """
    Parameters
    ----------
    dictionary : dict
        Dictionary obtained from get_dictionary() with the same keywords as run_tiempo() executed after this function. Input_dictionary in that function must be set to 'manual' or 'path'

    Returns
    -------
    None.

    Must be executed when the filter properties are changed.
    """
    if dictionary['savefolder'] == None:
        dictionary['savefolder'] = Path.cwd().joinpath('output_TiEMPO')
    else:
        dictionary['savefolder'] = convert_folder(dictionary['savefolder'])
    dictionary['sourcefolder'] = convert_folder(dictionary['sourcefolder'])
    dictionary = convert_grid(dictionary)
    
    length_EL_vector = 25
    length_pwv_vector = 25

    #test one value
    pwv_vector = np.logspace(0., 0., length_pwv_vector)
    EL_vector = np.linspace(60., 60., length_EL_vector)

    #make interpolation curves
    pwv_vector = np.logspace(-1, 0.35, length_pwv_vector)
    EL_vector = np.linspace(20., 90., length_EL_vector)
    
    ft1 = ft.filterbank(dictionary['F_min'], dictionary['spec_res'], dictionary['num_filters'], dictionary['f_spacing'], dictionary['num_bins'], dictionary['D1'])
    ft1.save_TP_data(EL_vector, pwv_vector)
    ft1.fit_TPpwvEL_curve(pwv_vector, EL_vector)
    ft1.save_etaF_data(pwv_vector, 90.)

def get_dictionary(input_dictionary, prefix_atm_data, sourcefolder, save_name_data, savefolder, save_P=True, save_T=True, n_jobs = 30, n_batches = 8,\
                   obs_time = 2., grid = .2, x_length_strip = 65536., separation = 1.1326,\
                   galaxy_on = True, luminosity = 13.7, redshift = 4.43, linewidth = 600, \
                   EL = 60, EL_vec = None, max_num_strips = 32, pwv_0 = 1., F_min = 220e9, \
                   num_bins = 1500, spec_res = 500, f_spacing = 500, \
                   num_filters = 347, beam_radius = 5., useDESIM = 1, \
                   inclAtmosphere = 1, windspeed = 10, D1 = 0, \
                   dictionary_name = ''):
    """
    Parameters
    ----------
    input_dictionary : string
        'deshima_1', 'deshima_2', 'manual' or 'path'. Determines where the input values of keywords F_min thru come from: either standard values for DESHIMA, manual entry from the keywords or from a txt file in the order of keywords obs_time thru D1.
    prefix_atm_data : string
        The prefix of the output of ARIS that is being used by the TiEMPO
    sourcefolder : string
        Folder where ARIS data used by the model is saved, relative to the current working directory. A parent folder can be specified by prefixing with '../', '..\\', './' or '.\\'
    save_name_data : string
        Prefix of the output of TiEMPO
    savefolder : string
        Folder where the output of the model will be saved relative to the current working directory. A parent folder can be specified by prefixing with '../', '..\\', './' or '.\\'
    save_P : bool
        determines whether the power in Watts is saved as an output. Default is True
    save_T : bool
        determines whether the sky temperature in Kelvins is saved as an output. Default is True.
    n_jobs : int
        maximum number of concurrently running jobs (size of thread-pool). -1 means all CPUs are used.
    n_batches : int
        number of batches the entire observation is divided up into. Default is 8.
    obs_time : float, optional
        Length of the observation in seconds. The default is 2..
    grid : float, optional
        The width of a grid square in the atmosphere map in meters. The default is .2.
    x_length_strip : int, optional
        The length of one atmosphere strip in the x direction in number of gridpoints (NOT METERS). The default is 65536.
    separation : float, optional
        Separation between two chop positions in m, assuming that the atmosphere is at 1km height. Default is 1.1326 (this corresponds to 116.8 arcsec).
    galaxy_on : bool, optional
        Determines whether there is a galaxy in position 2. The default is True.
    luminosity : float, optional
        Luminosity if the galaxy in log(L_fir [L_sol]). The default is 13.7.
    redshift : float, optional
        The redshift of the galaxy. The default is 4.43.
    linewidth : float, optional
        Width of the spectral lines in the galaxy spectrum in km/s. The default is 600.
    EL : float, optional
        Elevation of the telescope in degrees. The default is 60.
    EL_vec: vector of floats, optional
        If this parameter is set, it allows to specify the elevation of the telescope in degrees per timestep, for example in the case of tracking a target. Vector must have a length of 160Hz * obs_time.
    max_num_strips : int, optional
        Number of atmosphere strips that are saved in the ARIS output folder. The default is 32.
    pwv_0 : float, optional
        Baseline value of precipitable water vapor that is added to the d(pwv) from ARIS in mm. The default is 1.
    F_min : float, optional
        Lowest center frequency of all the MKIDs. The default is 220e9.
    num_bins : int, optional
         determines the amount of bins used in the simulation of the galaxy spectrum. The default is 1500.
    spec_res : float, optional
        Spectral resolution. The default is 500.
    f_spacing : float, optional
        spacing between center frequencies = F/dF (mean). The default is 500.
    num_filters : int, optional
        Number of filters in the filterbank. The default is 347.
    beam_radius : float, optional
        Radius of the Gaussian beam in meters. The default is 5.
    useDESIM : int, optional
        1 or 0. Determines whether the simple atmospheric model is used (0) or the more sophisticated desim simulation (1). The default is 1.
    inclAtmosphere : int, optional
        1 or 0. Determines whether the atmosphere is included in the simulation. The default is 1 (yes).
    windspeed: float, optional
        Sped of the wind in meters/second. The default is 10.
    D1 : int, optional
        1 or 0. Determines whether DESHIMA 1.0 is simulated. The default is 0.
    dictionary_name : string, optional
        name of a txt file in which the values of optional keywords are saved. prefix_atm_data, sourcefolder, save_name_data, savefolder, n_jobs, save_P, save_T and EL_vec must still be set outside the file. Only used when input_dictionary is set to 'path'. The default is ''. Order of the entries in the txt file must be: F_min, num_bins, spec_res, f_spacing, num_filters, beam_radius, useDESIM, inclAtmosphere, D1, time, grid, x_length_strip, galaxy_on, luminosity, redshift, linewidth, EL, max_num_strips, pwv_0, windspeed, n_batches.

    Returns
    -------
    dictionary : dict
        Dictionary with the above keywords.

    """
    if input_dictionary == 'deshima_1':
        dictionary = {
            'F_min': 332e9,
            'num_bins': 1500,
            'spec_res': 300,
            'f_spacing': 380,
            'num_filters': 49,
            'beam_radius': 5.,
            'useDESIM': 1,
            'inclAtmosphere': 1,
            'D1': 1
        }
    elif input_dictionary == 'deshima_2':
        dictionary = {
            'F_min': 220e9,
            'num_bins': 1500,
            'spec_res': 500,
            'f_spacing': 500,
            'num_filters': 347,
            'beam_radius': 5.,
            'useDESIM': 1,
            'inclAtmosphere': 1,
            'D1': 0
        }
    elif input_dictionary == 'manual':
        dictionary = {
            'F_min': F_min,
            'num_bins': num_bins,
            'spec_res': spec_res,
            'f_spacing': f_spacing,
            'num_filters': num_filters,
            'beam_radius': beam_radius,
            'useDESIM': useDESIM,
            'inclAtmosphere': inclAtmosphere,
            'D1': D1
        }
    else: 
        d = np.loadtxt(os.getcwd()+'\\' + dictionary_name, comments='#')
        dictionary = {
            'F_min': d[0],
            'num_bins': d[1],
            'spec_res': d[2],
            'f_spacing': d[3],
            'num_filters': d[4],
            'beam_radius': d[5],
            'useDESIM': d[6],
            'inclAtmosphere': d[7],
            'D1': d[8],
            'time': d[9],
            'grid':d[10],
            'x_length_strip':d[11],
            'separation':d[12],
            'galaxy_on':d[13],
            'luminosity':d[14],
            'redshift':d[15],
            'linewidth':d[16],
            'EL':d[17],
            'max_num_strips':d[18],
            'pwv_0':d[19],
            'windspeed':d[20],
            'n_batches':d[21],
            'save_P': save_P,
            'save_T': save_T,
            'prefix_atm_data':prefix_atm_data,
            'save_name_data':save_name_data,
            'n_jobs':n_jobs,
            'savefolder' : savefolder,
            'sourcefolder' : sourcefolder,
            'EL_vec': EL_vec
        }
        return dictionary
    dictionary['n_jobs'] = int(n_jobs)
    dictionary['time'] = obs_time
    dictionary['prefix_atm_data']= prefix_atm_data
    dictionary['grid']= grid
    dictionary['x_length_strip']= float(x_length_strip)
    dictionary['galaxy_on'] = galaxy_on
    dictionary['luminosity']= luminosity
    dictionary['redshift']= redshift
    dictionary['linewidth']= linewidth
    dictionary['EL']= EL
    dictionary['max_num_strips']= max_num_strips
    dictionary['save_name_data']= save_name_data
    dictionary['pwv_0'] = pwv_0
    dictionary['windspeed'] = windspeed
    dictionary['savefolder'] = savefolder
    dictionary['sourcefolder'] = sourcefolder
    dictionary['EL_vec'] = EL_vec
    dictionary['save_P'] = save_P
    dictionary['save_T'] = save_T
    dictionary['n_batches'] = int(n_batches)
    dictionary['separation'] = separation
    return dictionary

def run_tiempo(input_dictionary, prefix_atm_data, sourcefolder, save_name_data, savefolder = None, save_P=True, save_T=True, n_jobs = 30, n_batches = 8,\
                   obs_time = 3600., grid = .2, x_length_strip = 65536., separation = 1.1326,\
                   galaxy_on = True, luminosity = 13.7, redshift = 4.43, linewidth = 600, \
                   EL = 60, EL_vec=None, max_num_strips = 32, pwv_0 = 1., F_min = 220e9, \
                   num_bins = 1500, spec_res = 500, f_spacing = 500, \
                   num_filters = 347, beam_radius = 5., useDESIM = 1, \
                   inclAtmosphere = 1, windspeed = 10, D1 = 0, dictionary_name = ''):
    """
    Parameters
    ----------
    input_dictionary : string
        'deshima_1', 'deshima_2', 'manual' or 'path'. Determines where the input values of keywords F_min thru come from: either standard values for DESHIMA, manual entry from the keywords or from a txt file.
    prefix_atm_data : string
        The prefix of the output of ARIS that is being used by the TiEMPO
    sourcefolder : string
        Folder where ARIS data used by the model is saved, relative to the current working directory. A parent folder can be specified by prefixing with '../', '..\\', './' or '.\\'
    save_name_data : string
        Prefix of the output of TiEMPO
    savefolder : string
        Folder where the output of the model will be saved relative to the current working directory. A parent folder can be specified by prefixing with '../', '..\\', './' or '.\\'
    save_P : bool
        determines whether the power in Watts is saved as an output. Default is True
    save_T : bool
        determines whether the sky temperature in Kelvins is saved as an output. Default is True.
    n_jobs : int
        maximum number of concurrently running jobs (size of thread-pool). -1 means all CPUs are used.
    n_batches : int
        number of batches the entire observation is divided up into. Default is 8.
    obs_time : float, optional
        Length of the observation in seconds. The default is 2.0.
    grid : float, optional
        The width of a grid square in the atmosphere map in meters. The default is .2.
    x_length_strip : int, optional
        The length of one atmosphere strip in the x direction in number of gridpoints (NOT METERS). The default is 65536.
    separation : float, optional
        Separation between two chop positions in m, assuming that the atmosphere is at 1km height. Default is 1.1326 (this corresponds to 116.8 arcsec).
    galaxy_on : bool, optional
        Determines whether there is a galaxy in position 2. T The default is True.
    luminosity : float, optional
        Luminosity if the galaxy in log(L_fir [L_sol]). The default is 13.7.
    redshift : float, optional
        The redshift of the galaxy. The default is 4.43.
    linewidth : float, optional
        Width of the spectral lines in the galaxy spectrum in km/s. The default is 600.
    EL : float, optional
        Elevation of the telescope in degrees. The default is 60.
    EL_vec: vector of floats, optional
        If this parameter is set, it allows to specify the elevation of the telescope in degrees per timestep, for example in the case of tracking a target. Vector must have a length of 160Hz * obs_time.
    max_num_strips : int, optional
        Number of atmosphere strips that are saved in the ARIS output folder. The default is 32.
    pwv_0 : float, optional
        Baseline value of precipitable water vapor that is added to the d(pwv) from ARIS in mm. The default is 1.
    F_min : float, optional
        Lowest center frequency of all the MKIDs. The default is 220e9.
    num_bins : int, optional
         determines the amount of bins used in the simulation of the galaxy spectrum. The default is 1500.
    spec_res : float, optional
        Spectral resolution. The default is 500.
    f_spacing : float, optional
        spacing between center frequencies = F/dF (mean). The default is 500.
    num_filters : int, optional
        Number of filters in the filterbank. The default is 347.
    beam_radius : float, optional
        Radius of the Gaussian beam in meters. The default is 5.
    useDESIM : int, optional
        1 or 0. Determines whether the simple atmospheric model is used (0) or the more sophisticated desim simulation (1). The default is 1.
    inclAtmosphere : int, optional
        1 or 0.Determines whether the simple atmospheric model is used (0) or the more sophisticated desim simulation (1). The default is 1 (yes).
    windspeed: float, optional
        Sped of the wind in meters/second. The default is 10.
    D1 : int, optional
        1 or 0. Determines whether DESHIMA 1.0 is simulated. The default is 0.
    dictionary_name : string, optional
        name of a txt file in which the values of optional keywords are saved. prefix_atm_data, sourcefolder, save_name_data, savefolder, n_jobs, save_P, save_T and EL_vec must still be set outside the file. Only used when input_dictionary is set to 'path'. The default is ''. Order of the entries in the txt file must be: F_min, num_bins, spec_res, f_spacing, num_filters, beam_radius, useDESIM, inclAtmosphere, D1, time, grid, x_length_strip, luminosity, redshift, linewidth, EL, max_num_strips, pwv_0, windspeed, n_batches.
    
    Returns
    -------
    time_vector: array of floats
        Moments in time at which the signal is calculated.
    center_freq: array of floats
        Center frequencies of the MKIDs
    
    Saves '<cwd>/output_TiEMPO/<save_name_data>+P_X.npy' OR '<savefolder>/<save_name_data>+P_X.npy': numpy array of floats
        array of the power values of the signal in Watts. Dimensions are: [5 x #filters x #timesamples], as 5 pwv values are taken for each timesample
    Saves '<cwd>/output_TiEMPO/<save_name_data>+T_X.npy' OR '<savefolder>/<save_name_data>+T_X.npy': numpy array of floats
        array of the power values of the signal converted to sky temperature in Kelvins. Dimensions are: [5 x #filters x #timesamples], as 5 pwv values are taken for each timesample
    """
    dictionary = get_dictionary(input_dictionary, prefix_atm_data, sourcefolder,\
                                save_name_data, savefolder, save_P, save_T, n_jobs, n_batches, obs_time, grid, \
                                x_length_strip, separation, galaxy_on,luminosity, redshift, \
                                linewidth, EL, EL_vec, max_num_strips, pwv_0, F_min, \
                                num_bins, spec_res, f_spacing, num_filters, \
                                beam_radius, useDESIM, inclAtmosphere, \
                                windspeed, D1, dictionary_name)
    if dictionary['savefolder'] == None:
        dictionary['savefolder'] = Path.cwd().joinpath('output_TiEMPO')
    else:
        dictionary['savefolder'] = convert_folder(dictionary['savefolder'])
    dictionary['sourcefolder'] = convert_folder(dictionary['sourcefolder'])
    
    if round(dictionary['separation']/dictionary['grid']) != 1e-6*round(1e6*dictionary['separation']/dictionary['grid']):
        raise ValueError('The separation is not an integer multiple of the ARIS grid size. Consider changing the separation to {:.5f} m or {:.5f} m instead of {} m'.format(dictionary['grid']*np.floor(dictionary['separation']/dictionary['grid']), dictionary['grid']*np.ceil(dictionary['separation']/dictionary['grid']), dictionary['separation']))
    
    num_steps = dictionary['separation'] / (dictionary['windspeed']/160)
    if round(num_steps) != num_steps:
        raise ValueError('Separation is not an integer multiple of atmosphere distance per sample. Consider changing the windspeed to {} m/s or {} m/s instead of {} m/s'.format(dictionary['separation']*160/np.ceil(num_steps), dictionary['separation']*160/np.floor(num_steps), dictionary['windspeed']))
    
    max_obs_time = calcMaxObsTime(dictionary)
    if obs_time > max_obs_time:
        raise ValueError('obs_time must be smaller than: ', max_obs_time)
        
    if dictionary['n_jobs'] < 1:
        raise ValueError('Please set a number of threads greater than or equal to 1 in n_jobs.')
    
    if dictionary['n_batches'] < 1:
        raise ValueError('Please set a number of signal batches greater than or equal to 1 in n_batches.')
    
    st1 = st.signal_transmitter(dictionary)
    [time_vector, center_freq] = st1.transmit_signal_DESIM_multf_atm()
    return time_vector, center_freq