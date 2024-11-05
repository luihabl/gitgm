import numpy as np 
import matplotlib.pyplot as plt
from scipy.constants import m_e, e, pi, k, epsilon_0 as eps_0, mu_0 
from scipy.integrate import trapezoid, solve_ivp, odeint
from scipy.interpolate import interp1d

from util import load_csv, load_cross_section
from aux import pressure, maxwellian_flux_speed, u_B, A_eff, A_eff_1, SIGMA_I, R_ind, h_L

class GlobalModel:

    def __init__(self, config_dict):
        self.load_chemistry()
        self.load_config(config_dict)

    def load_chemistry(self):
        e_el, cs_el  = load_cross_section('cross-sections/Xe/Elastic_Xe.csv')
        e_ex, cs_ex  = load_cross_section('cross-sections/Xe/Excitation1_Xe.csv')
        e_iz, cs_iz  = load_cross_section('cross-sections/Xe/Ionization_Xe.csv')

        T = np.linspace(0.1 * e / k, 100 * e / k, 5000)
        k_el_array = self.rate_constant(T, e_el, cs_el, m_e)
        k_ex_array = self.rate_constant(T, e_ex, cs_ex, m_e)
        k_iz_array = self.rate_constant(T, e_iz, cs_iz, m_e)

        self.K_el = interp1d(T, k_el_array, fill_value=(k_el_array[0], k_el_array[-1]), bounds_error=True)
        self.K_ex = interp1d(T, k_ex_array, fill_value=(k_ex_array[0], k_ex_array[-1]), bounds_error=True)
        self.K_iz = interp1d(T, k_iz_array, fill_value=(k_iz_array[0], k_iz_array[-1]), bounds_error=True)
        
        self.E_iz = 12.127 * e
        self.E_ex = 11.6 * e

    # vvvv This is the rate constant model described in the paper, to use just uncomment this and comment the interpolation functions
    # def K_el(self, T): 
    #     return 3e-13 * T / T

    # def K_ex(self, T):
    #     T_eV = k * T / e
    #     return 1.93e-19 * T_eV**(-0.5) * np.exp(- self.E_ex / (e * T_eV)) * np.sqrt(8 * e * T_eV / (pi * m_e))
    
    # def K_iz(self, T):
    #     T_eV = k * T / e
    #     K_iz_1 = 1e-20 * ((3.97 + 0.643 * T_eV - 0.0368 * T_eV**2) * np.exp(- self.E_iz / (e * T_eV))) * np.sqrt(8 * e * T_eV / (pi * m_e))        
    #     K_iz_2 = 1e-20 * (- 1.031e-4 * T_eV**2 + 6.386 * np.exp(- self.E_iz / (e * T_eV))) * np.sqrt(8 * e * T_eV / (pi * m_e))
    #     return 0.5 * (K_iz_1 + K_iz_2)
    # ^^^^

    def rate_constant(self, T_k, E, cs, m):
        T = T_k * k / e
        n_temperature = T.shape[0]
        v = np.sqrt(2 * E * e / m)
        k_rate = np.zeros(n_temperature)
        for i in np.arange(n_temperature):
            a = (m / (2 * pi * e * T[i]))**(3/2) * 4 * pi
            f = cs * v**3 * np.exp(- m * v**2 / (2 * e * T[i])) 
            k_rate[i] = trapezoid(a*f, x=v)
        return k_rate

    def load_config(self, config_dict):

        # Geometry
        self.R = config_dict['R']
        self.L = config_dict['L']
        
        # Neutral flow
        self.m_i = config_dict['m_i']
        self.Q_g = config_dict['Q_g']
        self.beta_g = config_dict['beta_g']
        self.kappa = config_dict['kappa']

        # Ions
        self.beta_i = config_dict['beta_i']
        self.V_beam = config_dict['V_beam']

        # Electrical
        self.omega = config_dict['omega']
        self.N = config_dict['N']
        self.R_coil = config_dict['R_coil']
        self.I_coil = config_dict['I_coil']

        # Initial values
        self.T_e_0 = config_dict['T_e_0']
        self.n_e_0 = config_dict['n_e_0']
        self.T_g_0 = config_dict['T_g_0']
        self.n_g_0 = pressure(self.T_g_0, self.Q_g, 
                              maxwellian_flux_speed(self.T_g_0, self.m_i),
                              self.A_g) / (k * self.T_g_0)
    
    @property
    def A_g(self): return self.beta_g * pi * self.R**2

    @property
    def A_i(self): return self.beta_i * pi * self.R**2

    @property
    def V(self): return pi * self.R**2 * self.L 

    @property
    def A(self): return 2*pi*self.R**2 + 2*pi*self.R*self.L

    @property
    def v_beam(self): return np.sqrt(2 * e * self.V_beam / self.m_i)

    def flux_i(self, T_e, T_g, n_e, n_g):
        return h_L(n_g, self.L) * n_e * u_B(T_e, self.m_i)

    def thrust_i(self, T_e, T_g, n_e, n_g):
        return self.flux_i(T_e, T_g, n_e, n_g) * self.m_i * self.v_beam * self.A_i

    def j_i(self, T_e, T_g, n_e, n_g):
        return self.flux_i(T_e, T_g, n_e, n_g) * e

    def eval_property(self, func, y):
        prop = np.zeros(y.shape[0])
        for i in np.arange(y.shape[0]):
            T_e = y[i][0]
            T_g = y[i][1]
            n_e = y[i][2]
            n_g = y[i][3]
            prop[i] = func(T_e, T_g, n_e, n_g)

        return prop

    def P_loss(self, T_e, T_g, n_e, n_g):
        a = self.E_iz * n_e * n_g * self.K_iz(T_e)
        b = self.E_ex * n_e * n_g * self.K_ex(T_e)
        c = 3 * (m_e / self.m_i) * k * (T_e - T_g) * n_e * n_g * self.K_el(T_e)
        d = 7 * k * T_e * n_e * u_B(T_e, self.m_i) * A_eff(n_g, self.R, self.L) / self.V
        
        return a + b + c + d

    def P_abs(self, T_e, n_e, n_g):
        return R_ind(self.R, self.L, self.N, self.omega, n_e, n_g, self.K_el(T_e)) * self.I_coil**2 / (2 * self.V)

    def gas_heating(self, T_e, T_g, n_e, n_g):
        K_in = SIGMA_I * maxwellian_flux_speed(T_g, self.m_i)
        lambda_0 = self.R / 2.405 + self.L / pi
        # lambda_0 =np.sqrt((self.R / 2.405)**2 + (self.L / pi)**2)
        a = 3 * (m_e / self.m_i) * k * (T_e - T_g) * n_e * n_g * self.K_el(T_e)
        b = (1/4) * self.m_i * (u_B(T_e, self.m_i)**2) * n_e * n_g * K_in 
        c = self.kappa * (T_g - self.T_g_0) * self.A / (self.V * lambda_0)
        return a + b - c

    def particle_balance_e(self, T_e, T_g, n_e, n_g):
        a = n_e * n_g * self.K_iz(T_e)
        b = n_e * u_B(T_e, self.m_i) * A_eff(n_g, self.R, self.L) / self.V
        return a - b

    def particle_balance_g(self, T_e, T_g, n_e, n_g):
        a = self.Q_g /self.V
        b = n_e * u_B(T_e, self.m_i) * A_eff_1(n_g, self.R, self.L, self.beta_i) / self.V
        c = n_e * n_g * self.K_iz(T_e)
        d = (1/4) * n_g * maxwellian_flux_speed(T_g, self.m_i) * self.A_g / self.V
        return a + b - c - d

    def P_rf(self, T_e, T_g, n_e, n_g):
        R_ind_val = R_ind(self.R, self.L, self.N, self.omega, n_e, n_g, self.K_el(T_e))
        return (1/2) * (R_ind_val + self.R_coil) * self.I_coil**2

    def f_dy(self, t, y):
        T_e = y[0]
        T_g = y[1]
        n_e = y[2]
        n_g = y[3]

        particle_balance_e = self.particle_balance_e(T_e, T_g, n_e, n_g)
        particle_balance_g = self.particle_balance_g(T_e, T_g, n_e, n_g)

        dy = np.zeros(4)
        dy[0] = ((2 /(3 * k)) * (self.P_abs(T_e, n_e, n_g) - self.P_loss(T_e, T_g, n_e, n_g)) - T_e * particle_balance_e) / n_e
        dy[1] = ((2 /(3 * k)) * self.gas_heating(T_e, T_g, n_e, n_g) - T_g * particle_balance_g) / n_g
        dy[2] = particle_balance_e
        dy[3] = particle_balance_g 

        return dy

    def solve(self, t0, tf):
        y0 = np.array([self.T_e_0, self.T_g_0, self.n_e_0, self.n_g_0])
        return solve_ivp(self.f_dy, (t0, tf), y0, method='LSODA')


    def solve_for_I_coil(self, I_coil):
        p = np.zeros(I_coil.shape[0])
        solution = np.zeros((I_coil.shape[0], 4))

        for i, I in enumerate(I_coil):
            self.I_coil = I

            sol = self.solve(0, 5e-2)

            T_e = sol.y[0][-1]
            T_g = sol.y[1][-1]
            n_e = sol.y[2][-1]
            n_g = sol.y[3][-1]

            p[i] = self.P_rf(T_e, T_g, n_e, n_g)

            solution[i] = np.array([T_e, T_g, n_e, n_g])

        return p, solution
