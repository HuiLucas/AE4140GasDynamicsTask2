import numpy as np
from numba_kernels import (
    prandtl_meyer_nb,
    inverse_prandtl_meyer_nb,
    inverse_expansion_fan_function_nb,
    next_step_core_nb,
)


def approx_equal(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main():
    gamma = 1.4
    M = 2.0
    nu = prandtl_meyer_nb(M, gamma)
    M_back = inverse_prandtl_meyer_nb(nu, gamma)
    print("nu(M=2.0)", nu)
    print("M_back", M_back)
    assert approx_equal(M, M_back, 1e-6)

    nu0 = prandtl_meyer_nb(M, gamma)
    # target psi from a small phi
    phi_true = 0.05
    psi_target = phi_true - np.arcsin(1.0 / inverse_prandtl_meyer_nb(nu0 + phi_true, gamma))
    phi_sol = inverse_expansion_fan_function_nb(psi_target, gamma, M)
    print("phi_sol", phi_sol)
    assert approx_equal(phi_true, phi_sol, 1e-5)

    # Build a tiny synthetic previous column and mid arrays
    n_points = 6
    nozzle_exit_mach = M
    phi_edge = 0.1
    prev_y = np.linspace(0.0, 1.0, n_points)
    prev_x = np.zeros(n_points)
    prev_phi = np.linspace(0.0, phi_edge, n_points)
    prev_nu = np.ones(n_points) * nu0

    # Build mid arrays coherently similar to init_space_march
    prev_x2 = np.zeros(n_points - 1)
    prev_y2 = np.zeros(n_points - 1)
    prev_phi2 = np.zeros(n_points - 1)
    prev_nu2 = np.zeros(n_points - 1)

    for i in range(n_points - 1):
        phi_A = prev_phi[i]
        phi_B = prev_phi[i + 1]
        nu_A = prev_nu[i]
        nu_B = prev_nu[i + 1]
        M_a = inverse_prandtl_meyer_nb(nu_A, gamma)
        M_b = inverse_prandtl_meyer_nb(nu_B, gamma)
        mu_a = np.arcsin(1.0 / M_a)
        mu_b = np.arcsin(1.0 / M_b)
        prev_phi2[i] = 0.5 * (phi_A + phi_B + nu_B - nu_A)
        prev_nu2[i] = 0.5 * (nu_A + nu_B + phi_B - phi_A)
        M_p = inverse_prandtl_meyer_nb(prev_nu2[i], gamma)
        mu_p = np.arcsin(1.0 / M_p)
        a_A = 0.5 * (phi_A + prev_phi2[i] + mu_a + mu_p)
        a_B = 0.5 * (phi_B - mu_b + prev_phi2[i] - mu_p)
        denom = (np.tan(a_A) - np.tan(a_B))
        if denom == 0.0:
            denom = 1e-16
        prev_x2[i] = (prev_y[i + 1] - prev_y[i] + prev_x[i] * np.tan(a_A) - prev_x[i + 1] * np.tan(a_B)) / denom
        prev_y2[i] = prev_y[i] + (prev_x2[i] - prev_x[i]) * np.tan(a_A)

    out_x = np.empty(n_points)
    out_y = np.empty(n_points)
    out_phi = np.empty(n_points)
    out_nu = np.empty(n_points)
    out_M = np.empty(n_points)
    out_P = np.empty(n_points)
    out_x2 = np.empty(n_points - 1)
    out_y2 = np.empty(n_points - 1)
    out_phi2 = np.empty(n_points - 1)
    out_nu2 = np.empty(n_points - 1)
    out_M2 = np.empty(n_points - 1)
    out_P2 = np.empty(n_points - 1)

    stop = next_step_core_nb(
        n_points,
        gamma,
        0.11,  # next_phi_edge slightly larger
        0.0,   # use default custom nu
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
    )
    print("stop:", stop)
    # Basic sanity checks
    assert np.isfinite(out_x).all()
    assert np.isfinite(out_y).all()
    assert np.isfinite(out_phi).all()
    assert np.isfinite(out_nu).all()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()

