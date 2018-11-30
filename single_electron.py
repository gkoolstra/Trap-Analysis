import numpy as np
from scipy import optimize
from tabulate import tabulate

def argselectdomain(xdata, domain):
    ind = np.searchsorted(xdata, domain)
    return (ind[0], ind[1])

def selectdomain(xdata, ydata, domain):
    ind = np.searchsorted(xdata, domain)
    return xdata[ind[0]:ind[1]], ydata[ind[0]:ind[1]]

def voltage_to_frequency(V, V_start, V_crossing, f_cavity):
    """
    Converts voltage to frequency. Assumes a shape f = a np.sqrt(V - b)
    :param V: voltage in V
    :param Vstart: voltage at which frequency = 0
    :param Vcrossing: voltage at which the electron frequency crosses the resonator
    :param f_cavity: cavity frequency in Hz
    :return: Electron mode frequency estimate
    """
    return f_cavity * np.sqrt(np.abs(V - V_start)) / np.sqrt(V_crossing - V_start)

def voltage_to_frequency_from_derivative(V, V_crossing, f_cavity, derivative):
    """
    Converts electrode voltage to electron frequency using a the derivative at the crossing point.
    Assumes a shape f = a np.sqrt(V - b)
    :param V: voltage in V (array)
    :param V_crossing: voltage at which the electron frequency crosses the resonator
    :param f_cavity: cavity frequency in Hz
    :param derivative: slope of the frequency/voltage curve at V = V_crossing in units of Hz/V
    :return: Electron mode frequency estimate
    """
    b = V_crossing - f_cavity / (2 * derivative)
    output = np.sqrt(2 * derivative * f_cavity * (V - V_crossing) + f_cavity ** 2)

    if isinstance(V, np.ndarray):
        output[V < b] = 0
    else:
        if V < b:
            return 0

    return output

def susceptibility(g, f_probe, f_electron, gamma):
    """
    Returns the susceptibility of the hybrid system
    :param g: coupling in Hz
    :param f_probe: microwave probe frequency in Hz
    :param f_electron: electron mode frequency in Hz (can be obtained from voltage_to_frequency)
    :param gamma: electron dephasing rate in Hz
    :return: susceptibility
    """
    return (2 * np.pi * g) ** 2 / (2 * np.pi * (f_probe - f_electron) + 1j * 2 * np.pi * gamma)

def s21(kappa_tot, g, f_cavity, f_probe, f_electron, gamma):
    """
    Calculates the transmission, both phase and magnitude, of the cavity that is coupled to the electron.
    :param kappa_tot: f0 / Q_loaded (the coupling rate leaving through the left and right mirrors, assuming symmetric mirrors + internal loss inside the cavity) in Hz. This is the FWHM.
    :param g: coupling in Hz, when the electron is resonant.
    :param f_cavity: cavity frequency in Hz
    :param f_probe: microwave probe frequency in Hz
    :param f_electron: electron mode frequency in Hz (can be obtained from voltage_to_frequency)
    :param gamma: electron dephasing rate in Hz
    :return: magnitude (linear), phase (radians)
    """
    scaled_g = g * np.sqrt(f_cavity / f_electron)
    susc = susceptibility(scaled_g, f_probe, f_electron, gamma)
    single_response = 2 * np.pi * (kappa_tot / 2.) / (2 * np.pi * (f_probe - f_cavity) - susc + 1j * 2 * np.pi * (kappa_tot / 2.))

    phase_s21 = np.arctan2(-np.imag(single_response), np.real(single_response))
    mag_s21 = np.abs(single_response)

    return mag_s21, phase_s21 - np.pi / 2.

def phase_function(voltage, f_cavity, f_probe, kappa_tot, *parameters, **kwargs):
    """
    Returns the phase in degrees for a single tone trace where the electron stays in the ground state.
    :param voltage: voltage data
    :param f_cavity: cavity frequency in Hz
    :param f_probe: microwave probe frequency in Hz
    :param kappa_tot: f0 / Q_loaded (the coupling rate leaving through the left and right mirrors, assuming symmetric mirrors + internal loss inside the cavity) in Hz. This is the FWHM.
    :param parameters: fit parameters, either [g, gamma, V_crossing, derivative] or without the derivative (then see kwargs)
    :param kwargs: if derivative is not a fit parameter, specify derivative here: --> "derivative=..."
    :return: phase in degrees
    """
    if len(parameters) == 4:
        [g, gamma, V_crossing, derivative] = parameters
    elif len(parameters) == 3:
        [g, gamma, V_crossing] = parameters
        derivative = kwargs.get('derivative')

    f_electron = voltage_to_frequency_from_derivative(voltage, V_crossing=V_crossing, f_cavity=f_cavity,
                                                      derivative=derivative)
    magnitude, phase = s21(kappa_tot, g, f_cavity, f_probe, f_electron, gamma)
    return phase * 180 / np.pi

def magnitude_function(voltage, f_cavity, f_probe, *parameters, **kwargs):
    """
    Returns the magnitude in dB for a single tone trace where the electron stays in the ground state.
    :param voltage: voltage data
    :param f_cavity: cavity frequency in Hz
    :param f_probe: microwave probe frequency in Hz
    :param kappa_tot: f0 / Q_loaded (the coupling rate leaving through the left and right mirrors, assuming symmetric mirrors + internal loss inside the cavity) in Hz. This is the FWHM.
    :param parameters: fit parameters, either [g, gamma, V_crossing, derivative] or without the derivative (then see kwargs)
    :param kwargs: if derivative is not a fit parameter, specify derivative here: --> "derivative=..."
    :return: magnitude as function of voltage in dB
    """
    if len(parameters) == 5:
        # For if you know the slope, but not the total linewidth
        [g, gamma, V_crossing, kappa_tot, scaling] = parameters
        derivative = kwargs.get('derivative')
    if len(parameters) == 4:
        # For if you don't know the slope, but you do know the linewidth and scaling
        [g, gamma, V_crossing, derivative] = parameters
        kappa_tot = kwargs.get('kappa_tot')
        scaling = kwargs.get('scaling', 1.0)
    elif len(parameters) == 3:
        # For if you know the derivative, linewidth and scaling
        [g, gamma, V_crossing] = parameters
        derivative = kwargs.get('derivative')
        kappa_tot = kwargs.get('kappa_tot')
        scaling = kwargs.get('scaling', 1.0)

    f_electron = voltage_to_frequency_from_derivative(voltage, V_crossing=V_crossing, f_cavity=f_cavity,
                                                      derivative=derivative)
    magnitude, phase = s21(kappa_tot, g, f_cavity, f_probe, f_electron, gamma)
    return 20 * np.log10(scaling * magnitude)

def fit_phase(voltage, phase, f_cavity, f_probe, kappa_tot, fix_derivative=False, fitguess=None, parambounds=None, domain=None,
              verbose=True, **kwargs):
    """
    Fit a phase response of a single electron coupled to a cavity
    :param voltage: voltage data
    :param phase: phase data
    :param f_cavity: cavity frequency in Hz
    :param f_probe: microwave probe frequency in Hz
    :param V_start: voltage at which frequency is assumed to be  = 0
    :param kappa_tot: f0 / Q_loaded (the coupling rate leaving through the left and right mirrors, assuming symmetric mirrors + internal loss inside the cavity) in Hz. This is the FWHM.
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

    if fitguess is not None:
        startparams = fitguess
    elif fix_derivative:
        # g, gamma, V_crossing
        startparams = [5E6, 100E6, np.mean(voltage)]
    else:
        # g, gamma, V_crossing, derivative
        startparams = [5E6, 100E6, np.mean(voltage), 50E9]

    def phase_fit_function(voltage, *parameters):
        if fix_derivative:
            [g, gamma, V_crossing] = parameters
            derivative = fix_derivative
        else:
            [g, gamma, V_crossing, derivative] = parameters

        f_electron = voltage_to_frequency_from_derivative(voltage, V_crossing=V_crossing, f_cavity=f_cavity,
                                                          derivative=derivative)
        magnitude, phase = s21(kappa_tot, g, f_cavity, f_probe, f_electron, gamma)
        return phase * 180 / np.pi

    bestfitparams, covmatrix = optimize.curve_fit(phase_fit_function, fitdatax, fitdatay, startparams, bounds=parambounds, **kwargs)

    try:
        fitparam_errors = np.sqrt(np.diag(covmatrix))
    except:
        print(covmatrix)
        print("Error encountered in calculating errors on fit parameters. This may result from a very flat parameter space")

    if verbose:
        parnames = ["g", chr(915), "V_crossing", "df/dV"]
        print(tabulate(zip(parnames, bestfitparams, fitparam_errors), headers=["Parameter", "Value", "Std"],
                       tablefmt="fancy_grid", floatfmt=".3e", numalign="center", stralign='left'))

    return bestfitparams, fitparam_errors