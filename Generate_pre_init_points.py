import numpy as np
from scipy.interpolate import interp1d
from numba_kernels import prandtl_meyer_nb, inverse_prandtl_meyer_nb

def generate_pre_init_points(radius, phi_edge, init_column_x, init_column_y, n_points):
    top_y = np.max(init_column_y)
    top_x = np.max(init_column_x)
    points= []
    right_side = interp1d(init_column_x.squeeze(), init_column_y.squeeze(), kind='linear', fill_value="extrapolate")
    xlist = np.linspace(0, top_x, n_points)
    ylist = np.linspace(0, top_y, n_points)
    for x_r in xlist:
        for y_r in ylist:
            #print(x_r, y_r)
            if y_r > radius + np.tan(phi_edge) * x_r:
                continue
            y_boundary = right_side(x_r)
            if y_r > y_boundary:
                continue
            points.append((x_r, y_r))

    points = np.array(points)
    return points


def psi_of_phi(phi, M_nozzle, gamma):
    nu_nozzle = prandtl_meyer_nb(M_nozzle, gamma)

    return phi - np.arcsin(1/inverse_prandtl_meyer_nb(nu_nozzle + phi, gamma))
def phi_of_psi(psi, M_nozzle, gamma):
    nu_nozzle = prandtl_meyer_nb(M_nozzle, gamma)

    phi_initial_guess = psi + np.arcsin(1/M_nozzle)
    func_values = psi_of_phi(phi_initial_guess, M_nozzle, gamma) - psi

    phi_solution = phi_initial_guess - func_values * 0.1  # Simple fixed-point iteration step
    for _ in range(50):  # Iterate to refine the solution
        func_values = psi_of_phi(phi_solution, M_nozzle, gamma) - psi
        phi_solution -= func_values * 0.1

    return phi_solution

def get_xy_with_vars(points, phi_edge, M_nozzle, custom_nu_edge, radius, gamma):
    points_phi = np.concatenate((points, np.zeros((np.shape(points)[0], 1))), axis=1)
    points_nu = np.concatenate((points, np.zeros((np.shape(points)[0], 1))), axis=1)
    points_M = np.concatenate((points, np.zeros((np.shape(points)[0], 1))), axis=1)
    points_P_over_Pe = np.concatenate((points, np.zeros((np.shape(points)[0], 1))), axis=1)
    for i in range(np.shape(points)[0]):
        x = points[i, 0]
        y = points[i, 1]
        M2 = inverse_prandtl_meyer_nb(custom_nu_edge, gamma)
        mu_2 = np.arcsin(1./M2)
        if y < radius - 1/(np.sqrt(M_nozzle**2 - 1)) * x:
            phi = 0
            nu = prandtl_meyer_nb(M_nozzle, gamma)
            M = M_nozzle
            P_over_Pe = 1.0
        elif y < radius + np.tan(phi_edge - mu_2) * x:
            psi = np.arctan((y - radius) / x)
            phi = phi_of_psi(psi, M_nozzle, gamma)
            nu = phi + prandtl_meyer_nb(M_nozzle, gamma)
            M = inverse_prandtl_meyer_nb(nu, gamma)
            P_over_Pe = ((1 + 0.5 * (1.4 - 1) * M ** 2) / (
                1 + 0.5 * (1.4 - 1) * M_nozzle ** 2)) ** (-1.4 / (1.4 + 1))
        else:
            phi = phi_edge
            nu = custom_nu_edge
            M = inverse_prandtl_meyer_nb(nu, gamma)
            P_over_Pe = ((1 + 0.5 * (1.4 - 1) * M ** 2) / (
                1 + 0.5 * (1.4 - 1) * M_nozzle ** 2)) ** (-1.4 / (1.4 + 1))
        points_phi[i, 2] = phi
        points_nu[i, 2] = nu
        points_M[i, 2] = M
        points_P_over_Pe[i, 2] = P_over_Pe
        if nu > 10:
            print(nu, M)
    return points_phi, points_nu, points_M, points_P_over_Pe


if __name__ == "__main__":
    print(inverse_prandtl_meyer_nb(2, 1.4))
    print(prandtl_meyer_nb(2.0, 1.4))