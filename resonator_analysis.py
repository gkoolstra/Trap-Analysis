import numpy as np
from matplotlib import pyplot as plt
import trap_analysis, sympy
from tqdm import tqdm
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import mpmath
from scipy.interpolate import interp1d

try:
    from Common import kfit, common
except:
    print "Could not import kfit and common. Please do so manually."

def get_resonator_constants():
    """
    Returns a dictionary of resonator constants used in the calculations in this module.
    :return: {f0, Z0, Q, input power}
    """
    constants = {'f0' : 6.0E9, 'Z0' : 50.0, 'Q' : 10000, 'P' : -100}
    return constants

class EOMSolver:

    def __init__(self, use_FEM_data=True):
        self.physical_constants = trap_analysis.get_constants()
        self.resonator_constants = get_resonator_constants()
        self.x = sympy.Symbol('x')
        self.use_FEM_data = True if use_FEM_data else False

    def RF_efield_data(self, xeval):
        xdata, Udata = self.x_RF_FEM, self.U_RF_FEM
        Ex_RF = interp1d(xdata, Udata, kind='cubic')
        return Ex_RF(xeval)

    def DC_curvature_data(self, xeval):
        xdata, Udata = self.x_DC_FEM, self.V_DC_FEM
        DC_curv = interp1d(xdata, Udata, kind='cubic')
        return DC_curv(xeval)

    def RF_potential(self, xeval, *p):
        """
        RF potential: U_RF. This is the potential when +/- 0.5 V is applied to the resonator.
        :param xeval: x-point
        :param p: [a0, a1] --> a0 + a1 * x
        :return: a0 + a1 * x
        """
        a0, a1 = p
        f = implemented_function(sympy.Function('f'), lambda y: a0 + a1*y)
        U_RF = lambdify(self.x, f(self.x))
        return U_RF(xeval)

    def RF_efield(self, xeval, *p):
        """
        Derivative of RF_potential
        :param xeval:
        :param p: [a0, a1] --> diff(a0 + a1 * x)
        :return: diff(a0 + a1 * x)
        """
        a0, a1 = p
        return np.float(mpmath.diff(lambda y: a0 + a1*y, xeval))

    def DC_potential(self, xeval, *p):
        """
        :param xeval: x-point
        :param p: [a0, a1, a2] --> a0 + a1*(x-a2)**2
        :return: a0 + a1*(x-a2)**2
        """
        a0, a1, a2 = p
        f = implemented_function(sympy.Function('f'), lambda y: a0 + a1*(y-a2)**2)
        V_DC = lambdify(self.x, f(self.x))
        return V_DC(xeval)

    def DC_curvature(self, xeval, *p):
        """
        :param xeval: x-point
        :param p: [a0, a1, a2] --> diff(diff(a0 + a1 * (x-a2)**2))
        :return: a1
        """
        a0, a1, a2 = p
        return a1

    def setup_eom(self, electron_positions, dc_params, rf_params):
        """
        Set up the Matrix used for determining the electron frequency.
        :param electron_positions: Electron positions, stacked in a column
        :param dc_params: [a0, a1, a2] --> a0 + a1*(x-a2)**2
        :param rf_params: [a0, a1] --> a0 + a1*x
        :return: M^(-1) * K
        """
        c = self.physical_constants
        r = self.resonator_constants

        omega0 = 2*np.pi*r['f0']
        L = r['Z0']/omega0
        C = 1/(omega0**2 * L)

        num_electrons = np.shape(electron_positions)[1]
        xe, ye = np.array(electron_positions)

        # Set up the inverse of the mass matrix first
        diag_invM = 1/c['m_e'] * np.ones(num_electrons + 1)
        diag_invM[0] = 1/L
        invM = np.diag(diag_invM)

        # Set up the kinetic matrix next
        K = np.zeros(np.shape(invM))
        K[0,0] = 1/C # Bare cavity
        if self.use_FEM_data:
            K[1:,0] = K[0,1:] = c['e']/C * self.RF_efield_data(xe)
        else:
            K[1:,0] = K[0,1:] = c['e']/C * self.RF_efield(xe, *rf_params) # Coupling terms

        kij_plus = np.zeros((num_electrons, num_electrons))
        for idx in tqdm(range(num_electrons)):
            rij = np.sqrt((xe[idx]-xe)**2 + (ye[idx]-ye)**2)
            tij = np.arctan((ye[idx]-ye)/(xe[idx]-xe))
            kij_plus[idx,:] = 1/4. * c['e']**2/(4*np.pi*c['eps0']) * (1 + 3*np.cos(2*tij))/rij**3

        np.fill_diagonal(kij_plus, 0)

        if self.use_FEM_data:
            K_block = -kij_plus + np.diag(2*c['e']*self.DC_curvature_data(xe) + np.sum(kij_plus, axis=1))
        else:
            K_block = -kij_plus + np.diag(2*c['e']*self.DC_curvature(xe, *dc_params) + np.sum(kij_plus, axis=1))

        K[1:,1:] = K_block

        return np.dot(invM, K)

    def solve_eom(self, LHS):
        """
        Solves the eigenvalues and eigenvectors for the system of equations constructed with setup_eom()
        :param LHS: matrix product of M^(-1) K
        :return: Eigenvalues, Eigenvectors
        """
        EVals, EVecs = np.linalg.eig(LHS)
        return EVals, EVecs

    def plot_dc_potential(self, x, *p, **kwargs):
        """
        Plot the DC potential with parameters p
        :param x: x-points, unit: micron. May be None if use_FEM_data = True
        :param p: [a0, a1, a2] --> a0 + a1*(x-a2)**2
        :return: None
        """
        if self.use_FEM_data:
            x_interp = np.linspace(np.min(self.x_DC_FEM), np.max(self.x_DC_FEM), 1E4+1)
            plt.plot(self.x_DC_FEM, self.V_DC_FEM, 'o', **common.plot_opt('darkorange'))
            plt.plot(x_interp*1E6, self.DC_curvature_data(x_interp), '-k', label='Interpolated')
            plt.legend(loc=0)
            plt.title("DC curvature from FEM data")
        else:
            plt.plot(x*1E6, self.DC_potential(x, *p), '-', color='darkorange', **kwargs)
            plt.title("DC potential")

        plt.xlim(np.min(x*1E6), np.max(x*1E6))
        plt.xlabel('x ($\mu$m)')

    def plot_rf_potential(self, x, *p, **kwargs):
        """
        Plot the RF potential with parameters p
        :param x: x-points, unit: microns. May be None if use_FEM_data = True
        :param p: [a0, a1] --> a0 + a1*x
        :return: None
        """
        if self.use_FEM_data:
            x_interp = np.linspace(np.min(self.x_RF_FEM), np.max(self.x_RF_FEM), 1E4+1)
            plt.plot(self.x_RF_FEM, self.U_RF_FEM, 'o', **common.plot_opt('darkorange'))
            plt.plot(x_interp*1E6, self.DC_curvature_data(x_interp), '-k', label='Interpolated')
            plt.legend(loc=0)
            plt.title(r"$E_x$ from FEM data")
        else:
            plt.plot(x*1E6, self.RF_potential(x, *p), '-', color='lightblue', **kwargs)
            plt.title("RF potential")

        plt.xlim(np.min(x)*1E6, np.max(x)*1E6)
        plt.xlabel('x ($\mu$m)')