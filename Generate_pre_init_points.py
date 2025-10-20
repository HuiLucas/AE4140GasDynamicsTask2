import numpy as np
from scipy.interpolate import interp1d
from numba_functions import prandtl_meyer_nb, inverse_prandtl_meyer_nb, inverse_expansion_fan_function_nb

def generate_pre_init_points(radius, phi_edge, init_column_x, init_column_y, n_points):
    # Generate a grid of points within the expansion fan region and uniform nozzle flow region and uniform flow region behind expansion fan.
    top_y = np.max(init_column_y)
    top_x = np.max(init_column_x)
    points= []
    # Right side boundary of region
    right_side = interp1d(init_column_x.squeeze(), init_column_y.squeeze(), kind='linear', fill_value="extrapolate")
    # Generate grid:
    xlist = np.linspace(0, top_x, n_points)
    ylist = np.linspace(0, top_y, n_points)
    for x_r in xlist:
        for y_r in ylist:
            # Set upper boundary or region.
            if y_r > radius + np.tan(phi_edge) * x_r:
                continue
            # Set right side boundrary of region.
            y_boundary = right_side(x_r)
            if y_r > y_boundary:
                continue
            # Add point to list if within region.
            points.append((x_r, y_r))

    points = np.array(points)
    return points



def get_xy_with_vars(points, phi_edge, M_nozzle, custom_nu_edge, radius, gamma):
    # Get flow variables at each point in the grid.

    # Initialize arrays to hold variables
    points_phi = np.concatenate((points, np.zeros((np.shape(points)[0], 1))), axis=1)
    points_nu = np.concatenate((points, np.zeros((np.shape(points)[0], 1))), axis=1)
    points_M = np.concatenate((points, np.zeros((np.shape(points)[0], 1))), axis=1)
    points_P_over_Pe = np.concatenate((points, np.zeros((np.shape(points)[0], 1))), axis=1)

    # For every point on the grid:
    for i in range(np.shape(points)[0]):
        x = points[i, 0]
        y = points[i, 1]
        M2 = inverse_prandtl_meyer_nb(custom_nu_edge, gamma)
        mu_2 = np.arcsin(1./M2)

        # If inside uniform flow region before expansion fan
        if y < radius - 1/(np.sqrt(M_nozzle**2 - 1)) * x:
            phi = 0
            nu = prandtl_meyer_nb(M_nozzle, gamma)
            M = M_nozzle
            P_over_Pe = 1.0

        # If inside expansion fan region
        elif y < radius + np.tan(phi_edge - mu_2) * x:
            psi = np.arctan((y - radius) / x)
            phi = inverse_expansion_fan_function_nb(psi, gamma, M_nozzle)
            nu = phi + prandtl_meyer_nb(M_nozzle, gamma)
            M = inverse_prandtl_meyer_nb(nu, gamma)
            P_over_Pe = ((1 + 0.5 * (1.4 - 1) * M ** 2) / (
                1 + 0.5 * (1.4 - 1) * M_nozzle ** 2)) ** (-1.4 / (1.4 - 1))

        # If inside uniform flow region after expansion fan
        else:
            phi = phi_edge
            nu = custom_nu_edge
            M = inverse_prandtl_meyer_nb(nu, gamma)
            P_over_Pe = ((1 + 0.5 * (1.4 - 1) * M ** 2) / (
                1 + 0.5 * (1.4 - 1) * M_nozzle ** 2)) ** (-1.4 / (1.4 - 1))

        # Store variables
        points_phi[i, 2] = phi
        points_nu[i, 2] = nu
        points_M[i, 2] = M
        points_P_over_Pe[i, 2] = P_over_Pe
    return points_phi, points_nu, points_M, points_P_over_Pe


