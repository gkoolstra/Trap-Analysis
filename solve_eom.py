import os, time, json, sys, scipy
sys.path.append("/Users/gkoolstra/Documents/Code")
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from termcolor import cprint
from Common import common, kfit
from TrapAnalysis import trap_analysis, artificial_anneal as anneal
from BEMHelper import interpolate_slow
from scipy.interpolate import RectBivariateSpline
from scipy.signal import convolve2d
from .resonator_analysis import get_resonator_constants

class FullSolver:
    def __init__(self, settings_file):
        self.dc_interpolator = None
        self.rf_interpolator = None

        self.physical_constants = self.get_constants()
        self.resonator_constants = get_resonator_constants()

        with open(settings_file) as datafile:
            settings = json.load(datafile)

        self.simulation_name = settings['prelims']['simulation_name']
        self.include_screening = settings['electrostatics']['include_screening']
        self.helium_thickness = settings['electrostatics']['helium_thickness']
        self.screening_length = 2 * self.helium_thickness
        self.inspect_potentials = settings['prelims']['inspect_potentials']
        self.use_gradient = settings['minimizer']['use_gradient']
        self.gradient_tolerance = settings['minimizer']['gradient_tolerance']
        self.epsilon = settings['minimizer']['epsilon']
        if settings['electron_handling']['use_annealing']:
            self.trap_annealing_steps = [settings['electron_handling']['annealing_temperature']] * \
                                   settings['electron_handling'][
                                       'annealing_steps']
        else:
            self.trap_annealing_steps = []
        self.max_x_displacement = settings['electron_handling']['max_x_displacement']
        self.max_y_displacement = settings['electron_handling']['max_y_displacement']
        self.remove_unbound_electrons = settings['electron_handling']['remove_unbounded_electrons']
        self.remove_bounds = settings['electron_handling']['remove_bounds']
        self.show_final_result = settings['prelims']['show_final_result']
        self.create_movie = settings['prelims']['create_movie']
        self.dc_spline_smoothing = 0
        self.rf_spline_smoothing = settings['electrostatics']['rf_spline_smoothing']
        self.sigma_x = settings['electrostatics']['sigma_x']
        self.sigma_y = settings['electrostatics']['sigma_y']

        self.N_electrons = settings['initial_condition']["N_electrons"]
        self.N_rows = int(settings['initial_condition']['N_rows'])
        self.row_spacing = settings['initial_condition']['row_spacing']
        self.N_cols = settings['initial_condition']['N_cols']
        self.col_spacing = settings['initial_condition']['col_spacing']
        self.box_length = settings['electrostatics']['box_length']  # This is the box length from the simulation in Maxwell.
        self.inserted_res_length = settings['electrostatics'][
            'inserted_res_length']  # This is the length to be inserted in the Maxwell potential; units are in m
        self.inserted_trap_length = settings['electrostatics'][
            'inserted_trap_length']  # This is the length to be inserted in the Maxwell potential; units are in m

        self.save_path = settings["file_handling"]["save_path"]
        self.master_path = settings["file_handling"]["input_data_path"]
        self.sub_dir = time.strftime("%y%m%d_%H%M%S_{}".format(self.simulation_name))

        self.interpolator_xbounds = (-2.5, 2.5) # Units are microns
        self.interpolator_ybounds = (-3.0, 3.0) # Units are microns

        self.dc_interpolator_xpoints = 2000
        self.dc_interpolator_ypoints = 1001

        self.rf_interpolator_xpoints = 2000
        self.rf_interpolator_ypoints = 2 * 1001

    def set_dc_interpolator(self, vres, vtrap, vrg, vtg):
        t = trap_analysis.TrapSolver()

        # Evaluate all files in the following range.
        # xeval = np.linspace(-self.box_length * 1E6, self.box_length * 1E6, self.dc_interpolator_xpoints)
        xeval = np.linspace(self.interpolator_xbounds[0], self.interpolator_xbounds[1], self.dc_interpolator_xpoints)
        yeval = anneal.construct_symmetric_y(self.interpolator_ybounds[0], self.dc_interpolator_ypoints)

        dx = np.diff(xeval)[0] * 1E-6
        dy = np.diff(yeval)[0] * 1E-6

        x_eval, y_eval, output = anneal.load_data(self.master_path, xeval=xeval, yeval=yeval, mirror_y=True,
                                                  extend_resonator=False,
                                                  inserted_trap_length=self.inserted_trap_length * 1E6,
                                                  do_plot=False,
                                                  inserted_res_length=self.inserted_res_length * 1E6,
                                                  smoothen_xy=None)  # (0.40E-6, 10 * dy))

        self.output = output
        x_eval, y_eval, cropped_potentials = t.crop_potentials(output, ydomain=None, xdomain=None)
        coefficients = np.array([vres, vtrap, vrg, vtg])
        combined_potential = t.get_combined_potential(cropped_potentials, coefficients)
        # Note: x_eval and y_eval are 1D arrays that contain the x and y coordinates at which the potentials are
        # evaluated
        # Units of x_eval and y_eval are um
        self.dc_interpolator = RectBivariateSpline(x_eval * 1E-6, y_eval * 1E-6, -combined_potential.T,
                                                   kx=3, ky=3, s=self.dc_spline_smoothing)

    def update_dc_interpolator(self, vres, vtrap, vrg, vtg):
        t = trap_analysis.TrapSolver()
        x_eval, y_eval, cropped_potentials = t.crop_potentials(self.output, ydomain=None, xdomain=None)
        coefficients = np.array([vres, vtrap, vrg, vtg])
        combined_potential = t.get_combined_potential(cropped_potentials, coefficients)
        # Note: x_eval and y_eval are 1D arrays that contain the x and y coordinates at which the potentials are
        # evaluated
        # Units of x_eval and y_eval are um
        self.dc_interpolator = RectBivariateSpline(x_eval * 1E-6, y_eval * 1E-6, -combined_potential.T,
                                                   kx=3, ky=3, s=self.dc_spline_smoothing)


    def set_rf_interpolator(self):

        elements, nodes, elem_solution, bounding_box = anneal.load_dsp(os.path.join(self.master_path, "RFfield.dsp"))
        xdata, ydata, Udata = interpolate_slow.prepare_for_interpolation(elements, nodes, elem_solution)
        xeval = np.linspace(-self.box_length * 1E6, self.box_length * 1E6, self.rf_interpolator_xpoints)
        yeval = np.linspace(-4.0, 4.0, self.rf_interpolator_ypoints)

        # The following data assumes a full model (not symmetrically mirrored model)
        xinterp, yinterp, Uinterp = interpolate_slow.evaluate_on_grid(xdata, ydata, Udata, xeval=xeval, yeval=yeval,
                                                                      clim=(-2E5, 2E5), plot_axes='xy',
                                                                      cmap=plt.cm.Spectral_r, plot_mesh=False,
                                                                      plot_data=False)

        self.rf_interpolator = RectBivariateSpline(xeval * 1E-6, yeval * 1E-6, -Uinterp.T,
                                                   kx=3, ky=3, s=self.rf_spline_smoothing)

    def get_constants(self):
        """
        Returns a dictionary of physical constants used in the calculations in this module.
        :return: {'e' : 1.602E-19, 'm_e' : 9.11E-31, 'eps0' : 8.85E-12}
        """
        constants = {'e': 1.602E-19, 'm_e': 9.11E-31, 'eps0': 8.85E-12}
        return constants

    def Ex(self, xe, ye):
        return self.rf_interpolator.ev(xe, ye, dx=1)

    def Ey(self, xe, ye):
        return self.rf_interpolator.ev(xe, ye, dy=1)

    def gaussian_kernel(self, kernel_x, kernel_y, sigma_x, sigma_y):
        return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(
                - (kernel_x ** 2 / (2 * sigma_x ** 2) + kernel_y ** 2 / (2 * sigma_y ** 2)))

    def gaussian_xy_kernel(self, kernel_x, kernel_y, sigma_x, sigma_y):
        return kernel_x * kernel_y / (sigma_x ** 2 * sigma_y ** 2) * self.gaussian_kernel(kernel_x, kernel_y, sigma_x, sigma_y)

    def gaussian_xx_kernel(self, kernel_x, kernel_y, sigma_x, sigma_y):
        return (kernel_x ** 2 - sigma_x ** 2) * self.gaussian_kernel(kernel_x, kernel_y, sigma_x, sigma_y) / sigma_x ** 4

    def gaussian_yy_kernel(self, kernel_x, kernel_y, sigma_x, sigma_y):
        return (kernel_y ** 2 - sigma_y ** 2) * self.gaussian_kernel(kernel_x, kernel_y, sigma_x, sigma_y) / sigma_y ** 4

    # def curv_xx(self, xe, ye):
    #     return self.dc_interpolator.ev(xe, ye, dx=2)
    #
    # def curv_yy(self, xe, ye):
    #     return self.dc_interpolator.ev(xe, ye, dy=2)
    #
    # def curv_xy(self, xe, ye):
    #     return self.dc_interpolator.ev(xe, ye, dx=1, dy=1)

    def crop_potential(self, x, y, U, xrange, yrange):
        xmin_idx, xmax_idx = common.find_nearest(x, xrange[0]), common.find_nearest(x, xrange[1])
        ymin_idx, ymax_idx = common.find_nearest(y, yrange[0]), common.find_nearest(y, yrange[1])

        return x[xmin_idx:xmax_idx], y[ymin_idx:ymax_idx], U[ymin_idx:ymax_idx, xmin_idx:xmax_idx]

    def plot_gaussian_kernels(self):
        # Evaluate all files in the following range.
        xeval = np.linspace(-self.box_length * 1E6, self.box_length * 1E6, self.dc_interpolator_xpoints)
        yeval = anneal.construct_symmetric_y(-4.5, self.dc_interpolator_ypoints)

        dx = np.diff(xeval)[0] * 1E-6
        dy = np.diff(yeval)[0] * 1E-6

        k_x = np.arange(-5 * self.sigma_x, 5 * self.sigma_x + dx, dx)
        k_y = np.arange(-5 * self.sigma_y, 5 * self.sigma_y + dy, dy)

        kernel_x, kernel_y = np.meshgrid(k_x, k_y)

        fig = plt.figure(figsize=(8.,6.))
        common.configure_axes(13)
        plt.subplot(221)
        im = plt.pcolormesh(kernel_x * 1E6, kernel_y * 1E6,
                       self.gaussian_kernel(kernel_x, kernel_y, self.sigma_x, self.sigma_y) * dx * dy,
                       cmap=plt.cm.Reds, vmin=0)
        plt.ylabel("y (%sm)" % (chr(956)))
        plt.xlim(np.min(kernel_x) * 1E6, np.max(kernel_x) * 1E6)
        plt.ylim(np.min(kernel_y) * 1E6, np.max(kernel_y) * 1E6)
        plt.colorbar(im, fraction=0.046, pad=0.04)

        plt.subplot(222)
        im = plt.pcolormesh(kernel_x * 1E6, kernel_y * 1E6,
                       self.gaussian_xy_kernel(kernel_x, kernel_y, self.sigma_x, self.sigma_y) * dx * dy,
                       cmap=plt.cm.RdBu)
        plt.xlim(np.min(kernel_x) * 1E6, np.max(kernel_x) * 1E6)
        plt.ylim(np.min(kernel_y) * 1E6, np.max(kernel_y) * 1E6)
        plt.colorbar(im, fraction=0.046, pad=0.04)

        plt.subplot(223)
        im = plt.pcolormesh(kernel_x * 1E6, kernel_y * 1E6,
                       self.gaussian_xx_kernel(kernel_x, kernel_y, self.sigma_x, self.sigma_y) * dx * dy,
                       cmap=plt.cm.RdBu)
        plt.ylabel("y (%sm)" % (chr(956)))
        plt.xlabel("x (%sm)" % (chr(956)))
        plt.xlim(np.min(kernel_x) * 1E6, np.max(kernel_x) * 1E6)
        plt.ylim(np.min(kernel_y) * 1E6, np.max(kernel_y) * 1E6)
        plt.colorbar(im, fraction=0.046, pad=0.04)

        plt.subplot(224)
        im = plt.pcolormesh(kernel_x * 1E6, kernel_y * 1E6,
                       self.gaussian_yy_kernel(kernel_x, kernel_y, self.sigma_x, self.sigma_y) * dx * dy,
                       cmap=plt.cm.RdBu)
        plt.xlabel("x (%sm)" % (chr(956)))
        plt.xlim(np.min(kernel_x) * 1E6, np.max(kernel_x) * 1E6)
        plt.ylim(np.min(kernel_y) * 1E6, np.max(kernel_y) * 1E6)
        plt.colorbar(im, fraction=0.046, pad=0.04)

        fig.tight_layout()

    def setup_eom(self, ri):
        """
        Set up the Matrix used for determining the electron frequency.
        :param electron_positions: Electron positions, in the form [x0, y0, x1, y1, ...]
        :return: M^(-1) * K
        """
        c = self.physical_constants
        r = self.resonator_constants

        omega0 = 2*np.pi*r['f0']
        L = r['Z0']/omega0
        C = 1/(omega0**2 * L)

        num_electrons = int(len(ri)/2)
        xe, ye = anneal.r2xy(ri)

        # Set up the inverse of the mass matrix first
        diag_invM = 1/c['m_e'] * np.ones(2 * num_electrons + 1)
        diag_invM[0] = 1/L
        invM = np.diag(diag_invM)
        M = np.diag(np.array([L] + [c['m_e']]*(2 * num_electrons)))

        # Set up the kinetic matrix next
        Kij_plus, Kij_minus, Lij = np.zeros(np.shape(invM)), np.zeros(np.shape(invM)), np.zeros(np.shape(invM))
        K = np.zeros((2*num_electrons+1, 2*num_electrons+1))
        # Row 1 and column 1 only have bare cavity information, and cavity-electron terms
        K[0,0] = 1/C
        K[1:num_electrons+1,0] = K[0,1:num_electrons+1] = c['e']/C * self.Ex(xe, ye)
        K[num_electrons+1:2*num_electrons+1,0] = K[0,num_electrons+1:2*num_electrons+1] = c['e']/C * self.Ey(xe, ye)

        kij_plus = np.zeros((num_electrons, num_electrons))
        kij_minus = np.zeros((num_electrons, num_electrons))
        lij = np.zeros((num_electrons, num_electrons))

        Xi, Yi = np.meshgrid(xe, ye)
        Xj, Yj = Xi.T, Yi.T
        XiXj = Xi - Xj
        YiYj = Yi - Yj
        rij = np.sqrt((XiXj) ** 2 + (YiYj) ** 2)
        np.fill_diagonal(XiXj, 1E-15)
        tij = np.arctan(YiYj / XiXj)

        # Remember to set the diagonal back to 0
        np.fill_diagonal(tij, 0)
        # We'll be dividing by rij, so to avoid raising warnings:
        np.fill_diagonal(rij, 1E-15)

        if self.screening_length == np.inf:
            # print("Coulomb!")
            # Note that an infinite screening length corresponds to the Coulomb case. Usually it should be twice the
            # helium depth
            kij_plus = 1 / 4. * c['e'] ** 2 / (4 * np.pi * c['eps0']) * (1 + 3 * np.cos(2 * tij)) / rij ** 3
            kij_minus = 1 / 4. * c['e'] ** 2 / (4 * np.pi * c['eps0']) * (1 - 3 * np.cos(2 * tij)) / rij ** 3
            lij = 1 / 4. * c['e'] ** 2 / (4 * np.pi * c['eps0']) * 3 * np.sin(2 * tij) / rij ** 3
        else:
            # print("Yukawa!")
            rij_scaled = rij / self.screening_length
            kij_plus = 1 / 4. * c['e'] ** 2 / (4 * np.pi * c['eps0']) * np.exp(-rij_scaled) / rij ** 3 * \
                              (1 + rij_scaled + rij_scaled ** 2 + (3 + 3 * rij_scaled + rij_scaled ** 2) * np.cos(
                                  2 * tij))
            kij_minus = 1 / 4. * c['e'] ** 2 / (4 * np.pi * c['eps0']) * np.exp(-rij_scaled) / rij ** 3 * \
                              (1 + rij_scaled + rij_scaled ** 2 - (3 + 3 * rij_scaled + rij_scaled ** 2) * np.cos(
                                  2 * tij))
            lij = 1 / 4. * c['e'] ** 2 / (4 * np.pi * c['eps0']) * np.exp(-rij_scaled) / rij ** 3 * \
                              (3 + 3 * rij_scaled + rij_scaled ** 2) * np.sin(2 * tij)

        np.fill_diagonal(kij_plus, 0)
        np.fill_diagonal(kij_minus, 0)
        np.fill_diagonal(lij, 0)

        # Note: not sure where the factor 2 comes from
        Kij_plus = -kij_plus + np.diag(c['e']*self.curv_xx(xe, ye) + np.sum(kij_plus, axis=1))
        Kij_minus = -kij_minus + np.diag(c['e']*self.curv_yy(xe, ye) + np.sum(kij_minus, axis=1))
        Lij = -lij + np.diag(c['e']*self.curv_xy(xe, ye) + np.sum(lij, axis=1))

        K[1:num_electrons+1,1:num_electrons+1] = Kij_plus
        K[num_electrons+1:2*num_electrons+1, num_electrons+1:2*num_electrons+1] = Kij_minus
        K[1:num_electrons+1, num_electrons+1:2*num_electrons+1] = Lij
        K[num_electrons+1:2*num_electrons+1, 1:num_electrons+1] = Lij

        return K, M

    def solve_eom(self, LHS, RHS):
        """
        Solves the eigenvalues and eigenvectors for the system of equations constructed with setup_eom()
        :param LHS: matrix product of M^(-1) K
        :return: Eigenvalues, Eigenvectors
        """
        # EVals, EVecs = np.linalg.eig(np.dot(np.linalg.inv(RHS), LHS))
        EVals, EVecs = scipy.linalg.eigh(LHS, b=RHS)
        return EVals, EVecs

    def get_trap_electron_positions(self, Vres, Vtrap, Vrg, Vtg, N, initial_guess_x=None, initial_guess_y=None,
                                    use_adaptive_initial_guess=False, solve_equations_of_motion=True):

        Vcg = None
        electrode_names = ['resonator', 'trap', 'res_guard', 'trap_guard']
        if Vcg is not None:
            electrode_names.insert(3, 'ctr_guard')

        # Determine what electrode will be swept:
        sweep_points = np.inf

        for SweepIdx, Vpoint in enumerate([Vres, Vtrap, Vrg, Vcg, Vtg]):
            if isinstance(Vpoint, np.ndarray):
                sweep_points = Vpoint
                break

        if not (isinstance(sweep_points, np.ndarray)):
            sweep_points = [Vres]
            SweepIdx = 0

        cprint("Sweeping %s from %.2f V to %.2f V" % (electrode_names[SweepIdx], sweep_points[0], sweep_points[-1]),
               "green")

        center = (self.inserted_trap_length, 0E-6)
        radius = 0.5E-6
        noise_offset = 0.05E-6
        if initial_guess_x is None:
            init_trap_x = np.array(
                [center[0] + radius * np.cos(2 * np.pi * i / np.float(N)) for i in range(N)]) + np.random.normal(
                scale=noise_offset, size=N)
        else:
            assert len(initial_guess_x) == N
            init_trap_x = initial_guess_x
        if initial_guess_y is None:
            init_trap_y = np.array(
                [center[1] + radius * np.sin(2 * np.pi * i / np.float(N)) for i in range(N)]) + np.random.normal(
                scale=noise_offset, size=N)
        else:
            assert len(initial_guess_y) == N
            init_trap_y = initial_guess_y

        electron_initial_positions = anneal.xy2r(np.array(init_trap_x), np.array(init_trap_y))
        x_trap_init, y_trap_init = anneal.r2xy(electron_initial_positions)

        save = False

        # Evaluate all files in the following range.
        # xeval = np.linspace(-self.box_length * 1E6, self.box_length * 1E6, self.dc_interpolator_xpoints)
        xeval = np.linspace(self.interpolator_xbounds[0], self.interpolator_xbounds[1], self.dc_interpolator_xpoints)
        yeval = anneal.construct_symmetric_y(self.interpolator_ybounds[0], self.dc_interpolator_ypoints)

        dx = np.diff(xeval)[0] * 1E-6
        dy = np.diff(yeval)[0] * 1E-6

        x_eval, y_eval, output = anneal.load_data(self.master_path, xeval=xeval, yeval=yeval, mirror_y=True,
                                                  extend_resonator=False, inserted_trap_length=self.inserted_trap_length * 1E6,
                                                  do_plot=self.inspect_potentials,
                                                  inserted_res_length=self.inserted_res_length * 1E6,
                                                  smoothen_xy=None) #(0.40E-6, 10 * dy))

        if self.inspect_potentials:
            plt.show()

        # Note: x_eval and y_eval are 2D arrays that contain the x and y coordinates at which the potentials are evaluated
        conv_mon_save_path = os.path.join(self.save_path, self.sub_dir, "Figures")

        # This is where the actual solving starts...
        t = trap_analysis.TrapSolver()
        c = trap_analysis.get_constants()

        x_eval, y_eval, cropped_potentials = t.crop_potentials(output, ydomain=None, xdomain=None)
        electrons_in_the_trap = list()
        electron_positions = list(); energies = list()
        EVecs = list(); EVals = list()

        Second_Derivs = np.zeros((len(sweep_points), N, 3))

        for k, s in tqdm(enumerate(sweep_points)):
            coefficients = np.array([Vres[k], Vtrap[k], Vrg[k], Vcg, Vtg[k]])
            coefficients[SweepIdx] = s
            if Vcg is None:
                coefficients = np.delete(coefficients, 3)

            combined_potential = t.get_combined_potential(cropped_potentials, coefficients)
            # tilt_towards_trap = -5E-4 # units are eV/um. A positive tilt moves electron away from reservoir
            # tilt_array = np.reshape(np.tile(tilt_towards_trap * x_eval, len(y_eval)), (len(y_eval), len(x_eval))) # tilt point is centered around (x, y) = (0, 0) um
            # combined_potential -= tilt_array
            # Note: x_eval and y_eval are 1D arrays that contain the x and y coordinates at which the potentials are
            # evaluated
            # Units of x_eval and y_eval are um

            CMS = anneal.TrapAreaSolver(x_eval * 1E-6, y_eval * 1E-6, -combined_potential.T,
                                        spline_order_x=3, spline_order_y=3, smoothing=self.dc_spline_smoothing,
                                        include_screening=self.include_screening, screening_length=self.screening_length)

            # Define the second derivatives through convolution with a derivative of gaussians approach
            # kernel_x and kernel_y should have units of m
            kernel_x = np.arange(-5 * self.sigma_x, 5 * self.sigma_x + dx, dx)
            kernel_y = np.arange(-5 * self.sigma_y, 5 * self.sigma_y + dy, dy)

            # Test if we can construct the kernel accurately
            n_kernel_x = self.sigma_x / dx
            n_kernel_y = self.sigma_y / dy

            if n_kernel_x < 2 or n_kernel_y < 2:
                # Do not apply smoothing if the sampling of the potential is too course compared to sigma_x or sigma_y
                if k == 0:
                    print("(%s_x, %s_y) must be > (%.3f, %.3f) %sm to apply smoothing." % (chr(963), chr(963), 2 * dx * 1E6, 2 * dy * 1E6, chr(956)))
                    print("Ignoring smoothing conditions.")
                self.curv_xx = CMS.ddVdx
                self.curv_xy = CMS.ddVdxdy
                self.curv_yy = CMS.ddVdy
            else:
                # Apply gaussian smoothing and calculate the second derivative via convolution
                kernel_meshgrid = np.meshgrid(kernel_x, kernel_y)

                # Crop the data before convolving it.
                x_crop, y_crop, U_crop = self.crop_potential(x_eval, y_eval, -combined_potential,
                                                             xrange=(-1.5, 0), yrange=(-1.0, 1.0))

                # The approximations for the second derivatives are calculated by convolving the guassian kernels with the data
                g_xx = convolve2d(U_crop, self.gaussian_xx_kernel(kernel_meshgrid[0], kernel_meshgrid[1],
                                                                               self.sigma_x, self.sigma_y), mode='same')
                g_xy = convolve2d(U_crop, self.gaussian_xy_kernel(kernel_meshgrid[0], kernel_meshgrid[1],
                                                                               self.sigma_x, self.sigma_y), mode='same')
                g_yy = convolve2d(U_crop, self.gaussian_yy_kernel(kernel_meshgrid[0], kernel_meshgrid[1],
                                                                               self.sigma_x, self.sigma_y), mode='same')

                # Create the interpolating functions
                self.curv_xx = RectBivariateSpline(x_crop * 1E-6, y_crop * 1E-6, dx * dy * g_xx.T, kx=3, ky=3, s=0).ev
                self.curv_xy = RectBivariateSpline(x_crop * 1E-6, y_crop * 1E-6, dx * dy * g_xy.T, kx=3, ky=3, s=0).ev
                self.curv_yy = RectBivariateSpline(x_crop * 1E-6, y_crop * 1E-6, dx * dy * g_yy.T, kx=3, ky=3, s=0).ev


            X_eval, Y_eval = np.meshgrid(x_eval * 1E-6, y_eval * 1E-6)

            # Solve for the electron positions in the trap area!
            ConvMon = anneal.ConvergenceMonitor(Uopt=CMS.Vtotal, grad_Uopt=CMS.grad_total, N=1,
                                                Uext=CMS.V,
                                                xext=xeval * 1E-6, yext=yeval * 1E-6, verbose=False, eps=self.epsilon,
                                                save_path=None)

            ConvMon.figsize = (8., 2.)

            trap_minimizer_options = {'method': 'L-BFGS-B',
                                      'jac': CMS.grad_total,
                                      'options': {'disp': False, 'gtol': self.gradient_tolerance, 'eps': self.epsilon},
                                      'callback': None}

            # We save the initial Jacobian of the system for purposes.
            initial_jacobian = CMS.grad_total(electron_initial_positions)
            res = minimize(CMS.Vtotal, electron_initial_positions, **trap_minimizer_options)

            while res['status'] > 0:
                # Try removing unbounded electrons and restart the minimization
                if self.remove_unbound_electrons:
                    # Remove any electrons that are to the left of the trap
                    best_x, best_y = anneal.r2xy(res['x'])
                    idxs = np.where(np.logical_and(best_x > self.remove_bounds[0], best_x < self.remove_bounds[1]))[0]
                    best_x = np.delete(best_x, idxs)
                    best_y = np.delete(best_y, idxs)
                    # Use the solution from the current time step as the initial condition for the next timestep!
                    electron_initial_positions = anneal.xy2r(best_x, best_y)
                    if len(best_x) < len(res['x'][::2]):
                        print("%d/%d unbounded electrons removed. %d electrons remain." % (
                        np.int(len(res['x'][::2]) - len(best_x)), len(res['x'][::2]), len(best_x)))
                    res = minimize(CMS.Vtotal, electron_initial_positions, **trap_minimizer_options)
                else:
                    best_x, best_y = anneal.r2xy(res['x'])
                    idxs = np.union1d(np.where(best_x < -2E-6)[0], np.where(np.abs(best_y) > 2E-6)[0])
                    if len(idxs) > 0:
                        print("Following electrons are outside the simulation domain")
                        for i in idxs:
                            print("(x,y) = (%.3f, %.3f) um" % (best_x[i] * 1E6, best_y[i] * 1E6))
                    # To skip the infinite while loop.
                    break

            if res['status'] > 0:
                cprint("WARNING: Initial minimization for Trap did not converge!", "red")
                print("Final L-inf norm of gradient = %.2f eV/m" % (np.amax(res['jac'])))
                best_res = res
                if k == 0:
                    cprint("Please check your initial condition, are all electrons confined in the simulation area?", "red")
                    break

            if len(self.trap_annealing_steps) > 0:
                # cprint("SUCCESS: Initial minimization for Trap converged!", "green")
                # This maps the electron positions within the simulation domain
                cprint("Perturbing solution %d times at %.2f K. (dx,dy) ~ (%.3f, %.3f) um..." \
                       % (len(self.trap_annealing_steps), self.trap_annealing_steps[0],
                          np.mean(CMS.thermal_kick_x(res['x'][::2], res['x'][1::2], self.trap_annealing_steps[0],
                                                     maximum_dx=self.max_x_displacement)) * 1E6,
                          np.mean(CMS.thermal_kick_y(res['x'][::2], res['x'][1::2], self.trap_annealing_steps[0],
                                                     maximum_dy=self.max_y_displacement)) * 1E6),
                       "white")
                best_res = CMS.perturb_and_solve(CMS.Vtotal, len(self.trap_annealing_steps), self.trap_annealing_steps[0],
                                                 res, maximum_dx=self.max_x_displacement, maximum_dy=self.max_y_displacement,
                                                 **trap_minimizer_options)
            else:
                best_res = res

            if 0:
                electron_pos = best_res['x'][::2] * 1E6

                voltage_labels = "$V_\mathrm{res}$ = %.2f V\n$V_\mathrm{trap}$ = %.2f V\n$V_\mathrm{rg}$ = %.2f " \
                                 "V\n$V_\mathrm{tg}$ = %.2f V" \
                                 % (coefficients[0], coefficients[1], coefficients[2], coefficients[3])

                fig = plt.figure(figsize=(7., 3.))
                common.configure_axes(13)
                plt.plot(x_eval, CMS.V(x_eval * 1E-6, 0), '-', lw=0, color='orange', label=voltage_labels)
                plt.plot(x_eval, CMS.V(x_eval * 1E-6, 0), '-', color='orange')
                plt.plot(best_res['x'][::2] * 1E6, CMS.calculate_mu(best_res['x']), 'o', mec='none',
                         ms=4, color='cornflowerblue')
                plt.xlabel("$x$ ($\mu$m)")
                plt.ylabel("Potential energy (eV)")
                plt.title("%s = %.2f V" % (electrode_names[SweepIdx], coefficients[SweepIdx]))
                if k == 0:
                    lims = plt.ylim()
                plt.ylim(lims)
                plt.xlim(-7 + self.inserted_trap_length * 1E6, 7 + self.inserted_trap_length * 1E6)
                plt.legend(loc=1, numpoints=1, frameon=False, prop={'size': 10})
                plt.close(fig)

            x_plot = np.arange(-7E-6, +7E-6, dx) + self.inserted_trap_length
            y_plot = y_eval * 1E-6

            # Use the solution from the current time step as the initial condition for the next timestep!
            if use_adaptive_initial_guess:
                electron_initial_positions = best_res['x']
            ex, ey = anneal.r2xy(best_res['x'])
            electrons_in_the_trap.append(np.sum(np.logical_and(ex < self.inserted_trap_length + 1.5E-6,
                                                               ex > self.inserted_trap_length - 1.5E-6)))

            if 0:
                fig = plt.figure(figsize=(8., 3.))
                common.configure_axes(13)
                plt.plot(x_eval, CMS.V(x_eval * 1E-6, 0), '-k')
                plt.plot(ex * 1E6, CMS.calculate_mu(best_res['x']),
                         'o', color='violet', alpha=1)
                plt.xlabel("$x$ ($\mu$m)")
                plt.ylabel("Potential energy (eV)")
                plt.xlim(18, 28)
                plt.ylim(-0.35, -Vtrap[k] * 0.60)
                plt.vlines([self.inserted_trap_length*1E6 - 1.5, self.inserted_trap_length*1E6 + 1.5],
                           -0.35, -Vtrap * 0.60, linestyles='--', color='k')

            electron_positions.append(res['x'])
            energies.append(res['fun']) # this will contain the total minimized energy in eV

            if solve_equations_of_motion:
                LHS, RHS = self.setup_eom(best_res['x'])
                evals, evecs = self.solve_eom(LHS, RHS)
                EVals.append(evals)
                EVecs.append(evecs)

                Second_Derivs[k, :, 0] = self.curv_xx(ex, ey)
                Second_Derivs[k, :, 1] = self.curv_xy(ex, ey)
                Second_Derivs[k, :, 2] = self.curv_yy(ex, ey)

        trap_electrons_x, trap_electrons_y = anneal.r2xy(res['x'])

        if self.show_final_result:
            # Plot the resonator and trap electron configuration
            fig2 = plt.figure(figsize=(4, 3))
            common.configure_axes(13)
            plt.pcolormesh(x_eval, y_eval, CMS.V(X_eval, Y_eval), cmap=plt.cm.GnBu_r, vmax=0.0,
                           vmin=-0.75 * np.max(Vres))
            # plt.pcolormesh(x_eval, y_eval, CMS.V(X_eval, Y_eval), cmap=plt.cm.GnBu_r, vmax=0.0,
            #                vmin=-0.30)

            # This plots the initial condition, useful for debugging
            # plt.plot(x_trap_init * 1E6, y_trap_init * 1E6, 'o', color='mediumpurple', alpha=0.15)


            if best_res['status'] > 0:
                plt.text(self.inserted_trap_length*1E6, -2, "Minimization did not converge", fontdict={"size": 10})

            plt.xlabel("x ($\mu$m)")
            plt.ylabel("y ($\mu$m)")
            # plt.title("%d electrons" % (self.N_electrons))
            plt.xlim(self.inserted_trap_length*1E6 - 5, self.inserted_trap_length*1E6 + 5)
            plt.ylim(np.min(y_eval), np.max(y_eval))

            num_unbounded_electrons = anneal.check_unbounded_electrons(best_res['x'],
                                                                       xdomain=(np.min(x_eval) * 1E-6, np.max(x_eval) * 1E-6),
                                                                       ydomain=(np.min(y_eval) * 1E-6, np.max(y_eval) * 1E-6))

            anneal.draw_from_dxf(os.path.join(self.master_path, "all_electrodes.dxf"), offset=(0E-6, 0E-6),
                                 color="k", alpha=0.5)
            plt.plot(trap_electrons_x * 1E6, trap_electrons_y * 1E6, 'o', color='deeppink', alpha=1.0)
            print("Number of unbounded electrons = %d" % num_unbounded_electrons)
            plt.colorbar()
            plt.show()

            # fig2.tight_layout()
            # savepath2 = r"S:\Gerwin\Electron on helium\Papers\2017 - Circuit QED with a single electron on helium\Figure 3"
            # fig2.savefig(os.path.join(savepath2, "%d_electron_config.png" % N), dpi=300)

        return np.array(electron_positions), np.array(energies), np.array(EVecs), np.array(EVals), Second_Derivs