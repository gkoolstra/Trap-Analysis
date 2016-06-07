"""
Written by Gerwin Koolstra, Feb. 2016

Typical workflow:

* Load potentials
* Crop potentials
* Work with cropped potentials to investigate different combinations of coefficients
* Use fit_electron_potential and get_electron_frequency to find the electron frequency of a parabolic potential

"""
from tabulate import tabulate
import numpy as np
from matplotlib import pyplot as plt
from import_data import load_dsp, load_maxwell_data, select_domain

try:
    from Common import kfit, common
except:
    print "Could not import kfit and common. Please do so manually."

try:
    from BEM import interpolate_slow
except:
    print "Could not import BEM module. Please do so manually."

def get_constants():
    """
    Returns a dictionary of physical constants used in the calculations in this module.
    :return: {'e' : 1.602E-19, 'm_e' : 9.11E-31, 'eps0' : 8.85E-12}
    """
    constants = {'e' : 1.602E-19, 'm_e' : 9.11E-31, 'eps0' : 8.85E-12}
    return constants

class TrapSolver:
    """
    General methods:
    - load_potentials
    - crop_potentials
    - get_combined_potential
    - fit_electron_potential

    Methods assuming single electron:
    - get_electron_frequency
    - sweep_trap_coordinate
    - sweep_electrode_voltage
    - get_eigenfreqencies

    Methods assuming electron ensemble:
    -
    -
    """

    def __init__(self):
        self.__version__ = 1.0

    def load_potentials(self, fn_resonator, fn_currentloop, fn_resguard, fn_centerguard, fn_trapguard):
        """
        Load the 5 potential files.
        :param fn_resonator: Resonator filename
        :param fn_currentloop: Current loop filename
        :param fn_leftgate: Left gate filename
        :param fn_midgate: Center gate filename
        :param fn_rightgate: Right gate filename
        :return: Returns a list of dictionaries, each dictionary has keys 'V', 'x', and 'y'
        """
        output = list()
        potential_names = ['resonator', 'trap', 'resonatorguard', 'centerguard', 'trapguard']
        potential_fns = [fn_resonator, fn_currentloop, fn_resguard, fn_centerguard, fn_trapguard]
        for name, fn in zip(potential_names, potential_fns):
            if fn[-4:] == '.fld':
                x, y, V = load_maxwell_data(fn, do_log=False, figsize=(6.,5.), cmap=plt.cm.viridis)

                plt.title(name)

                output.append({'name' : name, 'V' : np.array(V, dtype=np.float64),
                               'x' : np.array(x, dtype=np.float64), 'y' : np.array(y, dtype=np.float64)})
            elif fn[-4:] == '.dsp':
                elements, nodes, elem_solution, bounding_box = load_dsp(fn)
                x, y, V = interpolate_slow.prepare_for_interpolation(elements, nodes, elem_solution)

                xcenter = np.mean(bounding_box[0:2])
                ycenter = np.mean(bounding_box[2:4])
                x -= xcenter
                y -= ycenter

                output.append({'name' : name, 'V' : np.array(V, dtype=np.float64),
                               'x' : np.array(x, dtype=np.float64), 'y' : np.array(y, dtype=np.float64)})

        return output

    def crop_potentials(self, potentials, xdomain=None, ydomain=None):
        """
        Crops the potentials. Useful if you want to crop the trap region, saves processing time.
        :param potentials: List of dictionaries.
        :param xdomain: Tuple. If set to None, there will be no cut.
        :param ydomain: Tuple. If set to None, there will be no cut.
        :return: x (2D array), y (2D array), List of cropped potentials
        """
        cropped_potentials = list()
        for p in potentials:
            x, y, V = select_domain(p['x'], p['y'], p['V'], xdomain=xdomain, ydomain=ydomain)
            cropped_potentials.append(V)

        return x, y, cropped_potentials

    def get_combined_potential(self, potentials, coefficients):
        """
        Sums the 5 different potentials up with weights specified in coefficients.
        :param potentials: List of potentials, as returned from load_potentials
        :param coefficients: List of coefficients
        :return: The summed potential
        """
        combined_potential = np.zeros(np.shape(potentials[0]))
        for c, p in zip(coefficients, potentials):
            combined_potential += c*p
        return combined_potential

    def fit_electron_potential(self, x, V, fitdomain=None, do_plot=False, plot_title=''):
        """
        Fits and plots (optional) the electron potential. Does not return the electron frequency.
        :param x: 1D array containing the position data, unit um.
        :param V: 1D array containing the potential data.
        :param fitdomain: Tuple
        :param do_plot: True/False to plot the data
        :param plot_title: Title that is given to the plot
        :return: Fitresult, Fiterrors in a list. List index is as follows: [offset, quadratic term, center]
        """
        # Fit the potential data to a parabola
        try:
            fr, ferr = kfit.fit_parabola(x, V, fitparams=[0, -1, 0], domain=fitdomain, verbose=False)
        except:
            raise ValueError("Fit error!")

        if fitdomain is not None:
            fitdatax, fitdatay = kfit.selectdomain(x, V, fitdomain)
        else:
            fitdatax, fitdatay = x, V

        # Get an estimate for the goodness of fit
        gof = kfit.get_rsquare(fitdatay, kfit.parabolafunc(fitdatax, *fr))

        if do_plot:
            plt.plot(x, V, '.k')
            plt.plot(fitdatax, kfit.parabolafunc(fitdatax, *fr), '-r', lw=2.0, label='$r^2$ = %.2f'%(gof))
            plt.xlabel("$x$ ($\mu$m)")
            plt.ylabel("Potential (V)")
            plt.xlim(min(x), max(x))
            plt.title(plot_title)
            #plt.legend(loc=0)

        return fr, ferr

    def get_electron_frequency(self, fr, ferr, verbose=True):
        """
        Gets the electron frequency from fit_electron_potential.
        :param fr: Fitresults (list). Curvature units for input are in V/um**2.
        :param ferr: Fiterrors (list).
        :param verbose: True/False, prints the frequency with standard deviation.
        :return: f (Hz), sigma_f (Hz)
        """
        # Calculate the electron frequency; Note that 0.5 ktrap x**2 = a * x**2
        c = get_constants()

        f = 1/(2*np.pi) * np.sqrt(-c['e']*2*fr[1]*1E12/c['m_e'])
        sigma_f = 1/(4*np.pi) * np.sqrt(-2*c['e']*1E12/(fr[1]*c['m_e'])) * ferr[1]

        if verbose:
            print "f = %.3f +/- %.3f GHz"%(f/1E9, sigma_f/1E9)

        return f, sigma_f

    def sweep_trap_coordinate(self, x, y, potentials, coefficients, sweep_data, sweep_coordinate='y',
                              fitdomain=(-0.5E-6, 0.5E-6), do_plot=False, print_report=False):
        """
        Sweep one of the coordinates of the trap to see if the minimum trap frequency occurs at the potential minimum.
        Electron frequency that is reported is the frequency of a single electron in the trap
        :param x: xdata, unit m
        :param y: ydata, unit m
        :param potentials: List of potentials
        :param coefficients: List of coefficients in the same order as potentials
        :param sweep_data: 1D array of x or y sweep points
        :param sweep_coordinate: 'x' or 'y'
        :param fitdomain: Tuple indicating the fit domain
        :param do_plot: Plot the electron frequency vs. sweep_data
        :param print_report: Prints a report with the electron frequency at each point in sweep_data
        :return:
        """
        V = self.get_combined_potential(np.array(potentials), np.array(coefficients))
        efreqs = list()
        efreqs_err = list()

        for s in sweep_data:
            if sweep_coordinate == 'y':
                yidx = common.find_nearest(y, s)
            else:
                xidx = common.find_nearest(x, s)

            if do_plot:
                plt.figure(figsize=(6.,4.))
                common.configure_axes(13)

            if sweep_coordinate == 'y':
                fr, ferr = self.fit_electron_potential(np.array(x, dtype=np.float64)*1E6, np.array(V[yidx,:], dtype=np.float64),
                                                  fitdomain=(fitdomain[0]*1E6, fitdomain[1]*1E6), do_plot=do_plot,
                                                  plot_title='Cutting in the %s direction'%sweep_coordinate)
            else:
                fr, ferr = self.fit_electron_potential(np.array(y, dtype=np.float64)*1E6, np.array(V[:,xidx], dtype=np.float64),
                                                  fitdomain=(fitdomain[0]*1E6, fitdomain[1]*1E6), do_plot=do_plot,
                                                  plot_title='Cutting in the %s direction'%sweep_coordinate)


            f, sigma_f = self.get_electron_frequency(fr, ferr, verbose=False)

            efreqs.append(f)
            efreqs_err.append(sigma_f)

        if print_report:
            print tabulate(zip(sweep_data*1E6, np.array(efreqs)/1E9, np.array(efreqs_err)/1E9),
                           headers=[sweep_coordinate+" (um)", 'f_min (GHz)', 'sigma_f (GHz)'],
                           tablefmt="rst", floatfmt=".3f", numalign="center", stralign='center')

        plt.figure()
        plt.title("Electron frequency vs. position inside trap")
        plt.errorbar(np.array(sweep_data)*1E6, np.array(efreqs)/1E9, yerr=np.array(efreqs_err)/1E9,
                     fmt='o', ecolor='red', **common.plot_opt('red'))
        plt.xlabel('%s ($\mu$m)'%(sweep_coordinate))
        plt.ylabel('$\omega_e/2\pi$ (GHz)')

    def sweep_electrode_voltage(self, x, y, potentials, coefficients, sweep_voltage, sweep_electrode_idx,
                                fitdomain=(-0.5E-6,+0.5E-6), clim=(-0.5, 0.5), do_plot=False, print_report=False,
                                f0=10E9, P=-80, Q=1E4, beta=0.243):
        """
        Sweep the voltage of one of the electrodes. Electron frequency reported is the frequency assuming there is a
        single electron in the trap.
        :param x: 1D array of position data in m.
        :param y: 1D array of position data in m.
        :param potentials: List of 5 potentials, as from crop_potentials()
        :param coefficients: List of 5 coefficients.
        :param sweep_voltage: Array of voltage points.
        :param sweep_electrode_idx: Integer indicating which element of potentials will be swept.
        :param fitdomain: Tuple with x coordinates in m.
        :param clim: Tuple that lets you control the scaling of the color plot of the potential landscape.
        :param do_plot: Plots a figure for every voltage point showing the potential landscape, fit in the minimum and center.
        :param print_repot: Prints a table showing the obtained frequencies in the center and at the minimum
        :return: f_minimum (Hz), sigmas_minimum, f_center (Hz), sigmas_center
        """

        f_minimum = list()
        sigmas_minimum = list()
        f_center = list()
        sigmas_center = list()
        evals = list()

        for voltage in sweep_voltage:
            coefficients[sweep_electrode_idx] = voltage
            V = self.get_combined_potential(np.array(potentials), np.array(coefficients))

            if do_plot:
                plt.figure(figsize=(18.,4.))
                plt.subplot(131)
                common.configure_axes(13)
                plt.title('Trap region zoom of combined potential')
                plt.pcolormesh(x*1E6, y*1E6, V, cmap=plt.cm.viridis, vmin=clim[0], vmax=clim[1])
                plt.xlim(np.min(x*1E6), np.max(x*1E6))
                plt.ylim(np.min(y*1E6), np.max(y*1E6))
                plt.colorbar()
                plt.xlabel('$x$ ($\mu$m)')
                plt.ylabel('$y$ ($\mu$m)')

            # Create a slice along x, where the combined potential is minimized
            yminidx = np.argmax(V)/np.shape(V)[1]
            yctridx = len(y)/2

            if do_plot:
                plt.plot(0, y[yctridx]*1E6, 'xr', alpha=0.5, ms=14)
                plt.plot(0, y[yminidx]*1E6, 'x', alpha=0.5, ms=14, color='white')

                plt.subplot(132)

            fr, ferr = self.fit_electron_potential(np.array(x[:]*1E6, dtype=np.float64), np.array(V[yminidx, :], dtype=np.float64),
                                              fitdomain=(fitdomain[0]*1E6, fitdomain[1]*1E6), do_plot=do_plot,
                                              plot_title='Minimum found for y = %.2f $\mu$m, V = %.2f V'%(y[yminidx]*1E6, voltage))

            f, sigma_f = self.get_electron_frequency(fr, ferr, verbose=False)

            evals.append(self.get_eigenfreqencies(-fr[1], beta, f0, P, Q))
            f_minimum.append(f)
            sigmas_minimum.append(sigma_f)

            # Create a slice along x, in the center of the trap
            if do_plot:
                plt.subplot(133)

            fr, ferr = self.fit_electron_potential(np.array(x[:]*1E6, dtype=np.float64), np.array(V[yctridx, :], dtype=np.float64),
                                              fitdomain=(fitdomain[0]*1E6, fitdomain[1]*1E6), do_plot=do_plot,
                                              plot_title='Center, V = %.2f V'%(voltage))

            f, sigma_f = self.get_electron_frequency(fr, ferr, verbose=False)

            f_center.append(f)
            sigmas_center.append(sigma_f)

        if print_report:
            print tabulate(zip(sweep_voltage, np.array(f_minimum)/1E9, np.array(f_center)/1E9),
                           headers=['V', 'f_min (GHz)', 'f_ctr (GHz)'],
                           tablefmt="rst", floatfmt=".3f", numalign="center", stralign='center')

        evals = np.array(evals)

        plt.figure(figsize=(12.,4.))
        plt.subplot(121)
        common.configure_axes(13)
        plt.errorbar(sweep_voltage, np.array(f_minimum)/1E9, yerr=np.array(sigmas_minimum)/1E9, fmt='o',
                     ecolor='black', label='Frequency at minimum', **common.plot_opt('black'))
        plt.errorbar(sweep_voltage, np.array(f_center)/1E9, yerr=np.array(sigmas_center)/1E9, fmt='o',
                     ecolor='blue', label='Frequency at center', **common.plot_opt('blue'))
        plt.xlabel('Potential')
        plt.ylabel('$\omega_e/2\pi$ (GHz)')
        plt.xlim(np.min(sweep_voltage), np.max(sweep_voltage))
        plt.legend(loc=0)

        plt.subplot(122)
        plt.plot(sweep_voltage*1E6, (evals[:,0]-f0)/1E6, '.-', color='black', label='Cavity response')
        plt.xlabel('Potential ($\mu$V)')
        plt.ylabel('$\Delta \omega_{\mathrm{cavity}}/2\pi$ (MHz)')
        plt.legend(loc=0)
        plt.xlim(np.min(sweep_voltage*1E6), np.max(sweep_voltage*1E6))

        return f_minimum, sigmas_minimum, f_center, sigmas_center

    def get_eigenfreqencies(self, alpha, beta, f0, P, Q):
        """
        Returns eigenfrequencies of the system of equations for a **single** electron - cavity system
        :param alpha: Quadratic component of the DC trapping potential in V/um**2
        :param beta: Linear component of the resonator differential mode potential in V/um.
        :param f0: Bare cavity frequency, without electrons, in Hz
        :param P: Power of the input drive in dBm
        :param Q: Q of the microwave cavity
        :return: 2 eigenvalues, one of which is the cavity frequency and one the electron frequency in Hz.
        """
        c = get_constants()
        omega0 = 2*np.pi*f0
        Z = 50.
        L = Z/omega0

        N = common.get_noof_photons_in_cavity(P, omega0/(2*np.pi), Q)
        voltage_scaling = np.sqrt(N) * np.sqrt(1.055E-34 * omega0**2 * Z/2.)
        a1 = alpha * 1E12 # Quadratic component of the DC potential in V/m**2
        beta = voltage_scaling * beta * 1E6 # Linear component of the RF potential in V/m

        Mat = np.array([[omega0**2, c['e'] * omega0**2 * beta],
                        [c['e'] * omega0**2 * L * beta/c['m_e'], 2 * c['e'] * a1/c['m_e']]])

        EVals = np.linalg.eigvals(Mat)

        return np.sqrt(EVals)/(2*np.pi)