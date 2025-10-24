import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def prandtl_meyer_nb(M, gamma):
    # NUMBA optimized Prandtl-Meyer function
    if M < 1.0:
        return 0.0
    if M < 1e6:
        gm1 = gamma - 1.0
        gp1 = gamma + 1.0
        t1 = np.sqrt(gp1/gm1)
        s = np.sqrt(max(M*M - 1.0, 0.0))
        t2 = np.arctan(np.sqrt((gm1/gp1) * (M*M - 1.0)))
        t3 = np.arctan(s)
        return t1 * t2 - t3
    else:
        return np.inf

@njit(cache=True, fastmath=True)
def inverse_prandtl_meyer_nb(nu_input, gamma, tol=1e-10, maximum_iterations=60):
    # NUMBA optimized inverse Prandtl-Meyer function using Newton-Raphson
    if nu_input < 0.0:
        return np.nan
    if abs(nu_input) == 0.0:
        return 1.0

    def PM_derivative(M):
        # derivative of Prandtl-Meyer function with respect to M
        if M <= 1.0:
            return 0.0
        return 2.0 * np.sqrt(max(M**2 - 1.0, 0.0)) / (M * ((gamma - 1.0) * M**2 + 2.0))

    # Newton-Raphson bounds
    M_lower = 1.0
    M_higher = 2.0
    while prandtl_meyer_nb(M_higher, gamma) < nu_input and M_higher < 1e8:
        M_higher *= 2.0

    nu_lower = 0.0
    nu_higher = prandtl_meyer_nb(M_higher, gamma)
    if nu_higher > nu_lower:
        M = M_lower + (nu_input - nu_lower) * (M_higher - M_lower) / (nu_higher - nu_lower)
    else:
        M = 1.0

    # Newton-Raphson iteration
    for _ in range(maximum_iterations):
        nu_M = prandtl_meyer_nb(M, gamma)
        remainder = nu_M - nu_input
        if abs(remainder) < tol:
            return M
        nu_der_eval = PM_derivative(M)
        if nu_der_eval != 0.0 and np.isfinite(nu_der_eval):
            M_new = M - remainder / nu_der_eval
        else:
            M_new = 0.5 * (M_lower + M_higher)
        if (M_new <= M_lower) or (M_new >= M_higher) or (M_new <= 1.0) or (not np.isfinite(M_new)):
            M_new = 0.5 * (M_lower + M_higher)
        if prandtl_meyer_nb(M_new, gamma) > nu_input:
            M_higher = M_new
        else:
            M_lower = M_new
        M = M_new
    return M

@njit(cache=True, fastmath=True)
def inverse_expansion_fan_function_nb(psi_input, gamma, mach_nozzle, tol=1e-10, maximum_iterations=120):
    # NUMBA optimized inverse expansion fan function using Newton-Raphson. Expansion fan function desribes relations between
    # flow deflection angle phi and - characterisitic angle psi in an expansion fan.
    nu_exit = prandtl_meyer_nb(mach_nozzle, gamma)
    phi = 0.0
    # Newton-Raphson iteration
    for _ in range(maximum_iterations):
        M = inverse_prandtl_meyer_nb(nu_exit + phi, gamma)
        if not np.isfinite(M) or M <= 1.0:
            M = max(M, 1.0000001)
        psi_val = phi - np.arcsin(1.0 / M)
        remainder = psi_val - psi_input
        if abs(remainder) < tol:
            return phi
        dnudM = 2.0 * np.sqrt(max(M**2 - 1.0, 0.0)) / (M * ((gamma - 1.0) * M**2 + 2.0))
        if dnudM == 0.0:
            dpsi = 1.0
        else:
            dMdphi = 1.0 / dnudM
            d_one_over_M_dphi = -(1.0 / (M**2)) * dMdphi
            denom = np.sqrt(max(1.0 - 1.0 / (M**2), 1e-30))
            dpsi = 1.0 - (1.0 / denom) * d_one_over_M_dphi
        if dpsi == 0.0 or not np.isfinite(dpsi):
            dpsi = 1.0
        phi -= remainder / dpsi
    return phi

@njit(cache=True, fastmath=True)
def next_step_core_nb(
    n_points,
    gamma,
    next_phi_edge,
    custom_nu_edge,
    nozzle_exit_mach,
    prev_x,
    prev_y,
    prev_phi,
    prev_nu,
    prev_x2,
    prev_y2,
    prev_phi2,
    prev_nu2,
    out_x,
    out_y,
    out_phi,
    out_nu,
    out_M,
    out_P,
    out_x2,
    out_y2,
    out_phi2,
    out_nu2,
    out_M2,
    out_P2,
):
    # NUMBA optimized core function for calculating the next column(s) in the method of characteristics.
    stop = False
    # Calculate properties at point at y=0 using the fact that phi=0 there.
    M_p0 = inverse_prandtl_meyer_nb(prev_nu2[0], gamma)
    mu_p0 = np.arcsin(1.0 / M_p0)
    out_x[0] = prev_x2[0] - prev_y2[0] / np.tan(prev_phi2[0] - mu_p0) # Position at intersection of y=0 and minus characteristic
    out_nu[0] = prev_nu2[0] + prev_phi2[0]
    out_phi[0] = 0.0
    out_M[0] = inverse_prandtl_meyer_nb(out_nu[0], gamma)
    out_P[0] = ((1.0 + 0.5 * (gamma - 1.0) * out_M[0] * out_M[0]) /
                (1.0 + 0.5 * (gamma - 1.0) * nozzle_exit_mach ** 2)) ** (-gamma / (gamma - 1.0))

    # Calculate properties at internal points by propagating from previous column
    for i in range(1, n_points - 1):
        # Using the defintions from the lecture
        phi_A = prev_phi2[i - 1]
        phi_B = prev_phi2[i]
        nu_A = prev_nu2[i - 1]
        nu_B = prev_nu2[i]
        M_a = inverse_prandtl_meyer_nb(nu_A, gamma)
        M_b = inverse_prandtl_meyer_nb(nu_B, gamma)
        mu_a = np.arcsin(1.0 / M_a)
        mu_b = np.arcsin(1.0 / M_b)
        out_phi[i] = 0.5 * (phi_A + phi_B + nu_B - nu_A)
        out_nu[i] = 0.5 * (nu_A + nu_B + phi_B - phi_A)
        M_p = inverse_prandtl_meyer_nb(out_nu[i], gamma)
        out_M[i] = M_p
        out_P[i] = ((1.0 + 0.5 * (gamma - 1.0) * out_M[i] * out_M[i]) /
                    (1.0 + 0.5 * (gamma - 1.0) * nozzle_exit_mach * nozzle_exit_mach)) ** (-gamma / (gamma - 1.0))
        mu_p = np.arcsin(1.0 / M_p)
        # Angles of the characteristic lines
        a_A = 0.5 * (phi_A + out_phi[i] + mu_a + mu_p)
        a_B = 0.5 * (phi_B - mu_b + out_phi[i] - mu_p)

        # Calculate position of point P by intersection of characteristic lines
        denominator = (np.tan(a_A) - np.tan(a_B))
        if denominator == 0.0:
            denominator = 1e-16
        out_x[i] = (prev_y2[i] - prev_y2[i - 1] + prev_x2[i - 1] * np.tan(a_A) - prev_x2[i] * np.tan(a_B)) / denominator
        out_y[i] = prev_y2[i - 1] + (out_x[i] - prev_x2[i - 1]) * np.tan(a_A)

        # Check for spacelike condition
        if (out_phi[i] + mu_p > np.pi * 0.5) or (out_phi[i] - mu_p < -np.pi * 0.5):
            stop = True
        # Check for shock condition
        #DDs = np.sqrt((out_x[i] - out_x[i-1])**2 + (out_y[i] - out_y[i-1])**2)
        # if (a_A > np.arctan((prev_y[i + 1] - prev_y[i]) / (prev_x[i + 1] - prev_x[i]))) and (prev_x[i+1] > 8):
        #     stop = True
        #     print('stopping because of schock')
        if out_y[i] < out_y[i-1]:
            if i > 5:
                if out_y[i] < out_y[i-5] and out_y[i]>0:
                    stop = True
                    print('stopping because of schock', i, out_y[i])

    # Calculate properties at the atmospheric boundary
    out_phi[n_points - 1] = next_phi_edge
    if custom_nu_edge == 0.0:
        out_nu[n_points - 1] = out_phi[n_points - 1] + prev_nu2[n_points - 2] - prev_phi2[n_points - 2]
    else:
        out_nu[n_points - 1] = custom_nu_edge
    out_M[n_points - 1] = inverse_prandtl_meyer_nb(out_nu[n_points - 1], gamma)
    out_P[n_points - 1] = ((1.0 + 0.5 * (gamma - 1.0) * out_M[n_points - 1] * out_M[n_points - 1]) /
                           (1.0 + 0.5 * (gamma - 1.0) * nozzle_exit_mach * nozzle_exit_mach)) ** (-gamma / (gamma - 1.0))

    # Define end point to be at the intersection of the + characteristic from the previous boundary point and the line going through
    # point -2 and -3 in the new column.
    dx_last = out_x[n_points - 2] - out_x[n_points - 3]
    if dx_last == 0.0:
        dx_last = 1e-16
    R = (out_y[n_points - 2] - out_y[n_points - 3]) / dx_last # Slope of the line through points -2 and -3 in new column
    tphi = np.tan(out_phi[n_points - 1]) # Slope of the + characteristic from previous boundary point
    denominator = (R - tphi)
    if denominator == 0.0:
        denominator = 1e-16
    out_x[n_points - 1] = (1.0 / denominator) * (
        prev_y[n_points - 1] - prev_x[n_points - 1] * tphi - out_y[n_points - 2] + out_x[n_points - 2] * R
    )
    out_y[n_points - 1] = prev_y[n_points - 1] + tphi * (out_x[n_points - 1] - prev_x[n_points - 1])

    # Calculate the _2 column points using the same method
    for i in range(n_points - 1):
        phi_A = out_phi[i]
        phi_B = out_phi[i + 1]
        nu_A = out_nu[i]
        nu_B = out_nu[i + 1]
        M_a = inverse_prandtl_meyer_nb(nu_A, gamma)
        M_b = inverse_prandtl_meyer_nb(nu_B, gamma)
        mu_a = np.arcsin(1.0 / M_a)
        mu_b = np.arcsin(1.0 / M_b)
        out_phi2[i] = 0.5 * (phi_A + phi_B + nu_B - nu_A)
        out_nu2[i] = 0.5 * (nu_A + nu_B + phi_B - phi_A)
        M_p = inverse_prandtl_meyer_nb(out_nu2[i], gamma)
        out_M2[i] = M_p
        out_P2[i] = ((1.0 + 0.5 * (gamma - 1.0) * out_M2[i] * out_M2[i]) /
                     (1.0 + 0.5 * (gamma - 1.0) * nozzle_exit_mach * nozzle_exit_mach)) ** (-gamma / (gamma - 1.0))
        mu_p = np.arcsin(1.0 / M_p)
        a_A = 0.5 * (phi_A + out_phi2[i] + mu_a + mu_p)
        a_B = 0.5 * (phi_B - mu_b + out_phi2[i] - mu_p)
        denominator = (np.tan(a_A) - np.tan(a_B))
        if denominator == 0.0:
            denominator = 1e-16
        out_x2[i] = (out_y[i + 1] - out_y[i] + out_x[i] * np.tan(a_A) - out_x[i + 1] * np.tan(a_B)) / denominator
        out_y2[i] = out_y[i] + (out_x2[i] - out_x[i]) * np.tan(a_A)
        if (out_phi2[i] + mu_p > np.pi * 0.5) or (out_phi2[i] - mu_p < -np.pi * 0.5):
            stop = True
            print('stopping because of spacelike')
        # if a_A > np.arctan((out_y[i+1] - out_y[i])/(out_x[i+1] - out_x[i])) and (out_x[i+1] > 8):
        #     stop = True
        #     print('stopping because of schock')
        if out_y2[i] < out_y2[i-1]:
            if i > 5:
                if out_y2[i] < out_y2[i-5] and out_y2[i]>0:
                    stop = True
                    print('stopping because of schock', i, out_y2[i])

    if np.max(out_x2) - np.min(out_x2) > 3:
        stop = True
        print('stopping because of numerical instability', np.max(out_x2), np.min(out_x2))

    return stop

