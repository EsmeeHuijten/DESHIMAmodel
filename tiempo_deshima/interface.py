#import main
import numpy as np
import signal_transmitter as st
import os

def calcMaxObsTime(dictionary):
    """
    Calculates the maximum observing time, using the number of gridpoints in
    one atmosphere strip, the gridwidth, the number of atmosphere strips and the
    windspeed
    """
    # maximum time
    # every strip has 32768 x-values
    separation = 2*0.5663 #gridpoints (116.8 arcseconds)
    max_obs_time = (dictionary['x_length_strip'] - 3 * separation)* \
                    dictionary['max_num_strips']*dictionary['grid'] \
                    /dictionary['windspeed']
    return max_obs_time

def get_dictionary(input_dictionary, prefix_atm_data, sourcefolder, save_name_data, savefolder, n_jobs,\
                   obs_time = 2., grid = .2, x_length_strip = 65536., \
                   luminosity = 13.7, redshift = 4.43, linewidth = 600, \
                   EL = 60, max_num_strips = 32, pwv_0 = 1., F_min = 220e9, \
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
    save_name_data : string
        Prefix of the output of TiEMPO
    n_jobs : int
        maximum number of concurrently running jobs (size of thread-pool). -1 means all CPUs are used.
    obs_time : float, optional
        Length of the observation in seconds. The default is 2..
    grid : float, optional
        The width of a grid square in the atmosphere map in meters. The default is .2.
    x_length_strip : int, optional
        The length of one atmosphere strip in the x direction in number of gridpoints (NOT METERS). The default is 65536..
    luminosity : float, optional
        Luminosity if the galaxy in log(L_fir [L_sol]). The default is 13.7.
    redshift : float, optional
        The redshift of the galaxy. The default is 4.43.
    linewidth : float, optional
        Width of the spectral lines in the galaxy spectrum in km/s. The default is 600.
    EL : float, optional
        Elevation of the telescope in degrees. The default is 60.
    max_num_strips : int, optional
        Numver of atmosphere strips that are saved in the ARIS output folder. The default is 32.
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
        name of a txt file in which the values of keywords obs_time thru D1 are saved. Only used when input_dictionary is set to 'path'. The default is ''. Order of the entries in the txt file must be: F_min, num_bins, spec_res, f_spacing, num_filters, beam_radius, useDESIM, inclAtmosphere, D1, time, grid, x_length_strip, luminosity, redshift, linewidth, EL, max_num_strips, pwv_0, windspeed.

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
            'luminosity':d[12],
            'redshift':d[13],
            'linewidth':d[14],
            'EL':d[15],
            'max_num_strips':d[16],
            'pwv_0':d[17],
            'windspeed':d[18],
            'prefix_atm_data':prefix_atm_data,
            'save_name_data':save_name_data,
            'n_jobs':n_jobs,
            'savefolder' : savefolder,
            'sourcefolder' : sourcefolder
        }
        return dictionary
    dictionary['n_jobs'] = n_jobs
    dictionary['time'] = obs_time
    dictionary['prefix_atm_data']= prefix_atm_data
    dictionary['grid']= grid
    dictionary['x_length_strip']= float(x_length_strip)
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
    return dictionary

def run_tiempo(input_dictionary, prefix_atm_data, sourcefolder, save_name_data, savefolder = None, n_jobs = 30,\
                   obs_time = 3600., grid = .2, x_length_strip = 65536., \
                   luminosity = 13.7, redshift = 4.43, linewidth = 600, \
                   EL = 60, max_num_strips = 32, pwv_0 = 1., F_min = 220e9, \
                   num_bins = 1500, spec_res = 500, f_spacing = 500, \
                   num_filters = 347, beam_radius = 5., useDESIM = 1, \
                   inclAtmosphere = 1, windspeed = 10, D1 = 0, dictionary_name = ''):
    """
    Parameters
    ----------
    input_dictionary : string
        'deshima_1', 'deshima_2', 'manual' or 'path'. Determines where the input values of keywords F_min thru come from: either standard values for DESHIMA, manual entry from the keywords or from a txt file in the order of keywords obs_time thru D1.
    prefix_atm_data : string
        The prefix of the output of ARIS that is being used by the TiEMPO
    save_name_data : string
        Prefix of the output of TiEMPO
    obs_time : float, optional
        Length of the observation in seconds. The default is 2..
    grid : float, optional
        The width of a grid square in the atmosphere map in meters. The default is .2.
    x_length_strip : int, optional
        The length of one atmosphere strip in the x direction in number of gridpoints (NOT METERS). The default is 65536..
    luminosity : float, optional
        Luminosity if the galaxy in log(L_fir [L_sol]). The default is 13.7.
    redshift : float, optional
        The redshift of the galaxy. The default is 4.43.
    linewidth : float, optional
        Width of the spectral lines in the galaxy spectrum in km/s. The default is 600.
    EL : float, optional
        Elevation of the telescope in degrees. The default is 60.
    max_num_strips : int, optional
        Numver of atmosphere strips that are saved in the ARIS output folder. The default is 32.
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
        name of a txt file in which the values of keywords obs_time thru D1 are saved. Only used when input_dictionary is set to 'path'. The default is ''.

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
                                save_name_data, savefolder, n_jobs, obs_time, grid, \
                                x_length_strip, luminosity, redshift, \
                                linewidth, EL, max_num_strips, pwv_0, F_min, \
                                num_bins, spec_res, f_spacing, num_filters, \
                                beam_radius, useDESIM, inclAtmosphere, \
                                windspeed, D1, dictionary_name)
    max_obs_time = calcMaxObsTime(dictionary)
    if obs_time > max_obs_time:
        raise ValueError('obs_time must be smaller than: ', max_obs_time)
    
    st1 = st.signal_transmitter(dictionary)
    [time_vector, center_freq] = st1.transmit_signal_DESIM_multf_atm()
    return time_vector, center_freq