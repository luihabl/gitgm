from scipy.constants import e, k, pi

config_dict = {

        # Geometry
        'R': 6e-2,
        'L': 10e-2,
        's': 1e-3,
        
        # Neutral flow
        'm_i': 2.18e-25,
        'Q_g': 1.2e19,
        'beta_g': 0.3,
        'kappa': 0.0057,

        # Ions
        'beta_i': 0.7,
        'V_beam': 1000,

        # Electrical
        'omega': 13.56e6 * 2 * pi,
        'N': 5,
        'R_coil': 2,
        'I_coil': 26,

        # Initial values
        'T_e_0': 3 * e / k,
        'n_e_0': 1e18,
        'T_g_0': 300

}

