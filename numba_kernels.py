import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def prandtl_meyer_nb(M, gamma):
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
def inverse_prandtl_meyer_nb(nu_target, gamma, tol=1e-10, maxiter=60):
    if nu_target < 0.0:
        return np.nan
    if abs(nu_target) == 0.0:
        return 1.0

    def dnu_dM(M):
        if M <= 1.0:
            return 0.0
        return 2.0 * np.sqrt(max(M*M - 1.0, 0.0)) / (M * ((gamma - 1.0) * M * M + 2.0))

    M_lo = 1.0
    M_hi = 2.0
    while prandtl_meyer_nb(M_hi, gamma) < nu_target and M_hi < 1e8:
        M_hi *= 2.0

    nu_lo = 0.0
    nu_hi = prandtl_meyer_nb(M_hi, gamma)
    if nu_hi > nu_lo:
        M = M_lo + (nu_target - nu_lo) * (M_hi - M_lo) / (nu_hi - nu_lo)
    else:
        M = 1.0

    for _ in range(maxiter):
        nu_M = prandtl_meyer_nb(M, gamma)
        f = nu_M - nu_target
        if abs(f) < tol:
            return M
        dfdM = dnu_dM(M)
        if dfdM != 0.0 and np.isfinite(dfdM):
            M_new = M - f / dfdM
        else:
            M_new = 0.5 * (M_lo + M_hi)
        if (M_new <= M_lo) or (M_new >= M_hi) or (M_new <= 1.0) or (not np.isfinite(M_new)):
            M_new = 0.5 * (M_lo + M_hi)
        if prandtl_meyer_nb(M_new, gamma) > nu_target:
            M_hi = M_new
        else:
            M_lo = M_new
        M = M_new
    return M

@njit(cache=True, fastmath=True)
def inverse_expansion_fan_function_nb(psi_target, gamma, mach_nozzle, tol=1e-10, maxiter=60):
    nu0 = prandtl_meyer_nb(mach_nozzle, gamma)
    phi = 0.0
    for _ in range(maxiter):
        M = inverse_prandtl_meyer_nb(nu0 + phi, gamma)
        if not np.isfinite(M) or M <= 1.0:
            M = max(M, 1.0000001)
        psi_val = phi - np.arcsin(1.0 / M)
        f = psi_val - psi_target
        if abs(f) < tol:
            return phi
        dnudM = 2.0 * np.sqrt(max(M*M - 1.0, 0.0)) / (M * ((gamma - 1.0) * M * M + 2.0))
        if dnudM == 0.0:
            dpsi = 1.0
        else:
            dMdphi = 1.0 / dnudM
            d_one_over_M_dphi = -(1.0 / (M * M)) * dMdphi
            denom = np.sqrt(max(1.0 - 1.0 / (M * M), 1e-30))
            dpsi = 1.0 - (1.0 / denom) * d_one_over_M_dphi
        if dpsi == 0.0 or not np.isfinite(dpsi):
            # fallback small step
            dpsi = 1.0
        phi -= f / dpsi
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
    stop = False
    M_p0 = inverse_prandtl_meyer_nb(prev_nu2[0], gamma)
    mu_p0 = np.arcsin(1.0 / M_p0)
    out_x[0] = prev_x2[0] - prev_y2[0] / np.tan(prev_phi2[0] - mu_p0)
    out_nu[0] = prev_nu2[0] + prev_phi2[0]
    out_phi[0] = 0.0
    out_M[0] = inverse_prandtl_meyer_nb(out_nu[0], gamma)
    out_P[0] = ((1.0 + 0.5 * (gamma - 1.0) * out_M[0] * out_M[0]) /
                (1.0 + 0.5 * (gamma - 1.0) * nozzle_exit_mach * nozzle_exit_mach)) ** (-gamma / (gamma + 1.0))


    for i in range(1, n_points - 1):
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
                    (1.0 + 0.5 * (gamma - 1.0) * nozzle_exit_mach * nozzle_exit_mach)) ** (-gamma / (gamma + 1.0))
        mu_p = np.arcsin(1.0 / M_p)
        a_A = 0.5 * (phi_A + out_phi[i] + mu_a + mu_p)
        a_B = 0.5 * (phi_B - mu_b + out_phi[i] - mu_p)
        denom = (np.tan(a_A) - np.tan(a_B))
        if denom == 0.0:
            denom = 1e-16
        out_x[i] = (prev_y2[i] - prev_y2[i - 1] + prev_x2[i - 1] * np.tan(a_A) - prev_x2[i] * np.tan(a_B)) / denom
        out_y[i] = prev_y2[i - 1] + (out_x[i] - prev_x2[i - 1]) * np.tan(a_A)
        if (out_phi[i] + mu_p > np.pi * 0.5) or (out_phi[i] - mu_p < -np.pi * 0.5):
            stop = True


    out_phi[n_points - 1] = next_phi_edge
    if custom_nu_edge == 0.0:
        out_nu[n_points - 1] = out_phi[n_points - 1] + prev_nu2[n_points - 2] - prev_phi2[n_points - 2]
    else:
        out_nu[n_points - 1] = custom_nu_edge
    out_M[n_points - 1] = inverse_prandtl_meyer_nb(out_nu[n_points - 1], gamma)
    out_P[n_points - 1] = ((1.0 + 0.5 * (gamma - 1.0) * out_M[n_points - 1] * out_M[n_points - 1]) /
                           (1.0 + 0.5 * (gamma - 1.0) * nozzle_exit_mach * nozzle_exit_mach)) ** (-gamma / (gamma + 1.0))


    dx_last = out_x[n_points - 2] - out_x[n_points - 3]
    if dx_last == 0.0:
        dx_last = 1e-16
    R = (out_y[n_points - 2] - out_y[n_points - 3]) / dx_last
    tphi = np.tan(out_phi[n_points - 1])
    denom = (R - tphi)
    if denom == 0.0:
        denom = 1e-16
    out_x[n_points - 1] = (1.0 / denom) * (
        prev_y[n_points - 1] - prev_x[n_points - 1] * tphi - out_y[n_points - 2] + out_x[n_points - 2] * R
    )
    out_y[n_points - 1] = prev_y[n_points - 1] + tphi * (out_x[n_points - 1] - prev_x[n_points - 1])


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
                     (1.0 + 0.5 * (gamma - 1.0) * nozzle_exit_mach * nozzle_exit_mach)) ** (-gamma / (gamma + 1.0))
        mu_p = np.arcsin(1.0 / M_p)
        a_A = 0.5 * (phi_A + out_phi2[i] + mu_a + mu_p)
        a_B = 0.5 * (phi_B - mu_b + out_phi2[i] - mu_p)
        denom = (np.tan(a_A) - np.tan(a_B))
        if denom == 0.0:
            denom = 1e-16
        out_x2[i] = (out_y[i + 1] - out_y[i] + out_x[i] * np.tan(a_A) - out_x[i + 1] * np.tan(a_B)) / denom
        out_y2[i] = out_y[i] + (out_x2[i] - out_x[i]) * np.tan(a_A)
        if (out_phi2[i] + mu_p > np.pi * 0.5) or (out_phi2[i] - mu_p < -np.pi * 0.5):
            stop = True
        # if i != n_points - 2:
        #     if (out_y2[i + 1] < out_y2[1]) or (out_y2[i + 1] < 0.0):
        #         stop = True
        if np.max(out_x2) - np.min(out_x2) > 20:
            stop = True

    return stop

