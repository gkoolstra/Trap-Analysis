import numpy as np
from scipy import optimize

def argselectdomain(xdata, domain):
    ind = np.searchsorted(xdata, domain)
    return (ind[0], ind[1])

def selectdomain(xdata, ydata, domain):
    ind = np.searchsorted(xdata, domain)
    return xdata[ind[0]:ind[1]], ydata[ind[0]:ind[1]]

def voltage_to_frequency(V, V_start, V_crossing, f_cavity):
    """
    Converts voltage to frequency
    :param V: voltage
    :param Vstart: voltage at which frequency = 0
    :param Vcrossing: voltage at which the electron frequency crosses the resonator
    :param f_cavity: cavity frequency in Hz
    :return: Electron mode frequency estimate
    """
    return f_cavity * np.sqrt(np.abs(V - V_start)) / np.sqrt(V_crossing - V_start)

def susceptibility(g, f_drive, f_electron, gamma):
    """
    Returns the susceptibility of the hybrid system
    :param g: coupling in Hz
    :param f_drive: microwave drive frequency in Hz
    :param f_electron: electron mode frequency in Hz (can be obtained from voltage_to_frequency)
    :param gamma: electron dephasing rate in Hz
    :return: susceptibility
    """
    return (2 * np.pi * g) ** 2 / (2 * np.pi * (f_drive - f_electron) + 1j * 2 * np.pi * gamma)

def s21(kappa_tot, g, f_cavity, f_drive, f_electron, gamma):
    """
    Calculates the transmission, both phase and magnitude, of the cavity that is coupled to the electron.
    :param kappa_tot: total linewidth (this is the sum of the input and output coupling rate)
    :param g: coupling in Hz
    :param f_cavity: cavity frequency in Hz
    :param f_drive: microwave drive frequency in Hz
    :param f_electron: electron mode frequency in Hz (can be obtained from voltage_to_frequency)
    :param gamma: electron dephasing rate in Hz
    :return: magnitude (linear), phase (radians)
    """
    susc = susceptibility(g, f_drive, f_electron, gamma)
    single_response = 2 * np.pi * kappa_tot / (2 * np.pi * (f_drive - f_cavity) - susc + 1j * 2 * np.pi * kappa_tot)

    phase_s21 = np.arctan2(-np.imag(single_response), np.real(single_response))
    mag_s21 = np.abs(single_response)

    return mag_s21, phase_s21 - np.pi / 2.

def fit_phase(voltage, phase, f_cavity, f_drive, V_start, kappa_cavity, fitguess=None, parambounds=None, domain=None, **kwargs):
    """
    Fit a phase response of a single electron coupled to a cavity
    :param voltage: voltage data
    :param phase: phase data
    :param f_cavity: cavity frequency in Hz
    :param f_drive: microwave drive frequency in Hz
    :param V_start: voltage at which frequency is assumed to be  = 0
    :param kappa_cavity: total line width
    :param fitguess: [g, gamma, V_crossing]
    :param parambounds: A list of bounds in the form ([lower1, lower2, lower3], [upper1, upper2, upper3])
    :param domain: tuple indicating the minimum and maximum of the voltage range to be taken into account
    :param kwargs: any additional arguments to be passed to scipy.optimize.curve_fit
    :return: bestfitparams, fitparam_errors
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(voltage, phase, domain)
    else:
        fitdatax = voltage
        fitdatay = phase

    if parambounds is None:
        parambounds = (-np.inf, +np.inf)

    # New in scipy 0.17:
    # * Parameter bounds: constrain fit parameters.
    #   Example: if there are 3 fit parameters which have to be constrained to (0, inf) we can use
    #   parambounds = ([0, 0, 0], [np.inf, np.inf, np.inf]), Alternatively, one may set: parambounds = (0, np.inf).
    #   Default is of course (-np.inf, np.inf)

    def phase_fit_function(voltage, *parameters):
        [g, gamma, V_crossing] = parameters
        f_electron = voltage_to_frequency(voltage, Vstart=V_start, Vcrossing=V_crossing, f_cavity=f_cavity)
        magnitude, phase = s21(kappa_cavity, g, f_cavity, f_drive, f_electron, gamma)
        return phase * 180 / np.pi

    if fitguess is not None:
        startparams = fitguess
    else:
        # g, gamma, V_crossing
        startparams = [5E6, 100E6, np.mean(voltage)]

    bestfitparams, covmatrix = optimize.curve_fit(phase_fit_function, fitdatax, fitdatay, startparams, bounds=parambounds, **kwargs)

    try:
        fitparam_errors = np.sqrt(np.diag(covmatrix))
    except:
        print(covmatrix)
        print("Error encountered in calculating errors on fit parameters. This may result from a very flat parameter space")

    return bestfitparams, fitparam_errors