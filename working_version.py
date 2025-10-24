import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
from scipy.interpolate import interp1d, NearestNDInterpolator

from Generate_pre_init_points import get_xy_with_vars, generate_pre_init_points
from numba_functions import prandtl_meyer_nb, inverse_prandtl_meyer_nb, inverse_expansion_fan_function_nb, next_step_core_nb
import copy

# Wrapper functions for the NUMBA-optimized functions as defined in numba_functions.py

def prandtl_meyer(M, gamma):
    return prandtl_meyer_nb(M, gamma)

def inverse_prandtl_meyer(nu_target, gamma, tol=1e-10, maxiter=60):
    return inverse_prandtl_meyer_nb(nu_target, gamma, tol, maxiter)

def inverse_expansion_fan_function(psi_target, gamma, mach_nozzle, tol=1e-10, maxiter=60):
    return inverse_expansion_fan_function_nb(psi_target, gamma, mach_nozzle, tol, maxiter)


# Definition of the column class, where a column represents a cross-section of the flow field composed of
# discrete points. At every propagation step, a new column is generated based on the Method of Characteristics as
# described in the lectures.

class column:
    def __init__(self, n_points):
        # initialize arrays to hold flow properties at discrete points in the column
        self.x_array = np.zeros((1,n_points))
        self.y_array = np.zeros((1,n_points))
        self.phi_array = np.zeros((1,n_points))
        self.nu_array = np.zeros((1,n_points))
        self.M_array = np.zeros((1,n_points))
        self.P_over_P_e_array = np.zeros((1,n_points)) # e denotes exit
        # the _2 arrays hold a column of points for which the endpoints are not the boundary of the flow field. As explained
        # in the lectures, there are altenating columns of points that either include the boundary or do not include the boundary.
        self.x_array_2 = np.zeros((1, n_points-1))
        self.y_array_2 = np.zeros((1, n_points-1))
        self.phi_array_2 = np.zeros((1, n_points-1))
        self.nu_array_2 = np.zeros((1, n_points-1))
        self.M_array_2 = np.zeros((1, n_points-1))
        self.P_over_P_e_array_2 = np.zeros((1, n_points-1))
        self.n_points = n_points
        self.nozzle_exit_mach = 0
        self.stop = False

    def init_space_march(self, radius, M_e, gamma, P_atmos_to_P_e=1.0):
        # Initialize the first column. This is NOT at the exit of the nozzle because of the singularity in characteristics at the corner
        # of the expansion fan. This causes small numerical deviations locally near that singularity to grow into large errors
        # further away. For this reason, this first column is located a small distance downstream of the nozzle exit.
        self.nozzle_exit_mach = M_e

        # Calculate nu and phi after the expansion fan, at the atmospheric boundary.
        nu_edge = prandtl_meyer(np.sqrt(
            (2 / (gamma - 1)) * (1 + 0.5 * (gamma - 1) * self.nozzle_exit_mach ** 2) * (P_atmos_to_P_e) ** (
                        (gamma - 1) / -gamma) - 2 / (gamma - 1)), gamma)
        self.nu_edge = nu_edge
        phi_edge = nu_edge - prandtl_meyer(self.nozzle_exit_mach, gamma)
        self.phi_edge = phi_edge

        mu_1 = np.arcsin(1 / self.nozzle_exit_mach) # at the exit
        mu_2 = np.arcsin(1 / inverse_prandtl_meyer(nu_edge, 1.4)) # at the atmospheric boundary

        # Position foot of the first column at intersection of first - characteristic and the middle line y=0.
        self.x_array[0, 0] = radius / np.tan(mu_1)
        self.y_array[0, 0] = 0.0
        self.phi_array[0, 0] = 0.0
        self.nu_array[0, 0] = prandtl_meyer(self.nozzle_exit_mach, gamma) + self.phi_array[0, 0]
        self.M_array[0, 0] = inverse_prandtl_meyer(self.nu_array[0, 0], gamma)
        self.P_over_P_e_array[0, 0] = ((1 + 0.5 * (gamma - 1) * self.M_array_2[0, 0] ** 2) / (
                1 + 0.5 * (gamma - 1) * self.nozzle_exit_mach ** 2)) ** (-gamma / (gamma - 1.))

        for i in range(1, n_vert):
            # If point is in the expansion fan:
            if i <= np.ceil((1 - mu_2 / (mu_1 + phi_edge)) * n_vert):
                # Psi is the angle between the upper corner of the nozzle exit and the point being calculated.
                psi = -mu_1 + i * (phi_edge - mu_2 + mu_1) / np.ceil((1 - mu_2 / (mu_1 + phi_edge)) * n_vert)
                # Psi of the point for i-1:
                prev_psi = psi - (phi_edge - mu_2 + mu_1) / np.ceil((1 - mu_2 / (mu_1 + phi_edge)) * n_vert)
                # Distance between point i-1 and i:
                Ds = np.cos(psi - np.arctan(
                    (self.x_array[0, i - 1] - 0) / (radius - self.y_array[0, i - 1]))) * np.sqrt(
                    (self.x_array[0, i - 1] - 0) ** 2 + (radius - self.y_array[0, i - 1]) ** 2) / np.cos(
                    inverse_expansion_fan_function(prev_psi, gamma, self.nozzle_exit_mach) - psi)
                # Local flow angle at point i-1:
                phi_local = inverse_expansion_fan_function(prev_psi, gamma, self.nozzle_exit_mach)
                # Calculate Dx and Dy components of Ds:
                Dy = Ds * np.cos(phi_local)
                Dx = -Dy * np.tan(phi_local)
                # Calculate coordinates of point i:
                self.x_array[0, i] = self.x_array[0, i - 1] + Dx
                self.y_array[0, i] = self.y_array[0, i - 1] + Dy
                # Calculate flow properties at point i:
                self.phi_array[0, i] = inverse_expansion_fan_function(psi, gamma, self.nozzle_exit_mach)
                self.nu_array[0, i] = prandtl_meyer(self.nozzle_exit_mach, gamma) + self.phi_array[
                    0, i]
                self.M_array[0, i] = inverse_prandtl_meyer(self.nu_array[0, i], gamma)
                self.P_over_P_e_array[0, i] = ((1 + 0.5 * (gamma - 1) * self.M_array[0, i] ** 2) / (
                            1 + 0.5 * (gamma - 1) * self.nozzle_exit_mach ** 2)) ** (-gamma / (gamma - 1.))
            # If point is in the uniform flow region after the expansion fan:
            else:
                # Calculate Psi at point i:
                Dpsi = (mu_2) / (n_vert - np.ceil((1 - mu_2 / (mu_1 + phi_edge)) * n_vert) - 1)
                psi = phi_edge - mu_2 + (i - np.ceil((1 - mu_2 / (mu_1 + phi_edge)) * n_vert)) * Dpsi
                # Calculate Ds between point i-1 and i:
                if psi <= 0.:
                    Ds = np.cos(psi - np.arctan(
                        (self.x_array[0, i - 1] - 0) / (radius - self.y_array[0, i - 1]))) * np.sqrt(
                        (self.x_array[0, i - 1] - 0) ** 2 + (
                                    radius - self.y_array[0, i - 1]) ** 2) / np.cos(phi_edge - psi)
                else:
                    Ds = np.sin(psi - np.arctan(
                        (self.y_array[0, i - 1] - radius) / (self.x_array[0, i - 1] - 0))) * np.sqrt(
                        (self.x_array[0, i - 1] - 0) ** 2 + (
                                    self.y_array[0, i - 1] - radius) ** 2) / np.cos(-phi_edge + np.arctan(
                        (self.y_array[0, i - 1] - radius) / (self.x_array[0, i - 1] - 0)))
                # Calculate Dx and Dy components of Ds:
                Dy = Ds * np.cos(phi_edge)
                Dx = -Dy * np.tan(phi_edge)
                # Calculate coordinates of point i:
                self.x_array[0, i] = self.x_array[0, i - 1] + Dx
                self.y_array[0, i] = self.y_array[0, i - 1] + Dy
                # Calculate flow properties at point i:
                self.phi_array[0, i] = phi_edge
                self.nu_array[0, i] = prandtl_meyer(self.nozzle_exit_mach, gamma) + phi_edge
                self.M_array[0, i] = inverse_prandtl_meyer(self.nu_array[0, i], gamma)
                self.P_over_P_e_array[0, i] = ((1 + 0.5 * (gamma - 1) * self.M_array[0, i] ** 2) / (
                            1 + 0.5 * (gamma - 1) * self.nozzle_exit_mach ** 2)) ** (-gamma / (gamma - 1.))
        # Calculate the _2 column properties by propogating one step:
        for i in range(self.n_points - 1):
            # Using the defitions from the lecture:
            phi_A = self.phi_array[0, i]
            phi_B = self.phi_array[0, i + 1]
            nu_A = self.nu_array[0, i]
            nu_B = self.nu_array[0, i + 1]
            M_a = inverse_prandtl_meyer(nu_A, gamma)
            M_b = inverse_prandtl_meyer(nu_B, gamma)
            mu_a = np.arcsin(1 / M_a)
            mu_b = np.arcsin(1 / M_b)
            self.phi_array_2[0, i] = 0.5 * (phi_A + phi_B + nu_B - nu_A)
            self.nu_array_2[0, i] = 0.5 * (nu_A + nu_B + phi_B - phi_A)
            M_p = inverse_prandtl_meyer(self.nu_array_2[0, i], gamma)
            mu_p = np.arcsin(1 / M_p)
            # Angles of the characteristic lines:
            a_A = 0.5 * (phi_A + self.phi_array_2[0, i] + mu_a + mu_p)
            a_B = 0.5 * (phi_B - mu_b + self.phi_array_2[0, i] - mu_p)
            # Calculate position of point P using intersection of characteristic lines:
            self.x_array_2[0, i] = (self.y_array[0, i + 1] - self.y_array[0, i] +
                                           self.x_array[0, i] * np.tan(a_A) -
                                           self.x_array[0, i + 1] * np.tan(a_B)) / (np.tan(a_A) - np.tan(a_B))
            self.y_array_2[0, i] = self.y_array[0, i] + (
                        self.x_array_2[0, i] - self.x_array[0, i]) * np.tan(a_A)
            self.M_array_2[0, i] = M_p
            self.P_over_P_e_array_2[0, i] = ((1 + 0.5 * (gamma - 1) * self.M_array_2[0, i] ** 2) / (
                    1 + 0.5 * (gamma - 1) * self.nozzle_exit_mach ** 2)) ** (-gamma / (gamma - 1))


    def next_step(self, gamma, next_phi_edge, custom_nu_edge=0):
        # Wrapper function of the next_step_core_nb function to generate the next column using NUMBA.
        next_column = column(self.n_points)
        next_column.nozzle_exit_mach = self.nozzle_exit_mach

        prev_x = self.x_array[0]
        prev_y = self.y_array[0]
        prev_phi = self.phi_array[0]
        prev_nu = self.nu_array[0]
        prev_x2 = self.x_array_2[0]
        prev_y2 = self.y_array_2[0]
        prev_phi2 = self.phi_array_2[0]
        prev_nu2 = self.nu_array_2[0]

        out_x = next_column.x_array[0]
        out_y = next_column.y_array[0]
        out_phi = next_column.phi_array[0]
        out_nu = next_column.nu_array[0]
        out_M = next_column.M_array[0]
        out_P = next_column.P_over_P_e_array[0]
        out_x2 = next_column.x_array_2[0]
        out_y2 = next_column.y_array_2[0]
        out_phi2 = next_column.phi_array_2[0]
        out_nu2 = next_column.nu_array_2[0]
        out_M2 = next_column.M_array_2[0]
        out_P2 = next_column.P_over_P_e_array_2[0]

        stop = next_step_core_nb( self.n_points, gamma, next_phi_edge, float(custom_nu_edge), next_column.nozzle_exit_mach, prev_x, prev_y, prev_phi, prev_nu, prev_x2, prev_y2, prev_phi2, prev_nu2, out_x, out_y, out_phi, out_nu, out_M, out_P, out_x2, out_y2, out_phi2, out_nu2, out_M2, out_P2)
        # Stop variable determines whether the propagation should stop, based on whether the next column becomes spacelike or has a numerical instability.
        next_column.stop = bool(stop)
        return next_column


# Initialize list of columns
columns = []
# Set number of points per column:
n_vert = 1000
# Initialize first column a small distance downstream of the nozzle exit.
init_column = column(n_vert)
# Set radius of the nozzle exit, exit Mach number, and pressure ratio between atmospheric pressure and exit pressure.
radius = 1
P_atmos_to_P_e = 0.5
M_exit = 2
gamma = 1.4
# Load the right initial column parameters, and append to list.
init_column.init_space_march(radius, M_exit, gamma, P_atmos_to_P_e)
columns.append(init_column)

# Set counter and list of flow direction angles at the boundary between flow field and atmosphere.
counter=0
phi_edges = []

# Add the positions of the column points to an array for plotting later.
xy = np.concatenate((init_column.x_array_2.T, init_column.y_array_2.T), axis=1)
xy = np.concatenate((xy, np.concatenate((init_column.x_array.T, init_column.y_array.T), axis=1)), axis=0)
xy_init = copy.deepcopy(xy)



stopstop = False # Stop variable for while loop
while stopstop == False and counter < 4*n_vert:
    counter += 1
    # Calculate flow direction angle at the atmospheric boundary for the next column.
    phi_edge = columns[-1].phi_array_2[0,-1] - columns[-1].nu_array_2[0,-1] + prandtl_meyer(np.sqrt((2/(gamma-1)) * (1 + 0.5*(gamma-1)*columns[-1].nozzle_exit_mach**2)*(P_atmos_to_P_e)**((gamma-1)/-gamma)-2/(gamma-1)), gamma)
    phi_edges.append(phi_edge)
    print(counter, phi_edge, columns[-1].stop, stopstop)
    # Append new column to list and add its points to the array for plotting later.
    new_column = columns[-1].next_step(gamma, phi_edge)
    xy = np.concatenate((xy, np.concatenate((new_column.x_array_2.T, new_column.y_array_2.T), axis=1)), axis=0)
    xy = np.concatenate((xy, np.concatenate((new_column.x_array.T, new_column.y_array.T), axis=1)), axis=0)
    if counter % 20 == 0:
        xy_init = np.concatenate((xy_init, np.concatenate((new_column.x_array.T, new_column.y_array.T), axis=1)), axis=0)

    # Stopping procedure:
    if new_column.stop == False:
        columns.append(new_column)

    else:
        stopstop = True
        xy_init = np.concatenate((xy_init, np.concatenate((new_column.x_array.T, new_column.y_array.T), axis=1)), axis=0)


# Plot columns:
fig0, ax0 = plt.subplots(dpi=600, figsize = (8,2))
ax0.set_aspect('equal')
plt.plot(xy_init[:,0], xy_init[:,1], ',r', markersize=0.1)
plt.title("Some of the columns used for computation")

# Generate arrays of (x,y) points with associated flow properties for interpolation and plotting.
print('1')
xy_points_with_phi = np.array(sum([[[col.x_array[0,i], col.y_array[0,i], col.phi_array[0,i]] for i in range(np.shape(col.x_array)[1])]
                               + [[col.x_array_2[0,i], col.y_array_2[0,i], col.phi_array_2[0,i]] for i in range(np.shape(col.x_array_2)[1])]

                               for col in columns], []))
print('2')
xy_points_with_nu = np.array(sum([[[col.x_array[0,i], col.y_array[0,i], col.nu_array[0,i]] for i in range(np.shape(col.x_array)[1])]
                               + [[col.x_array_2[0,i], col.y_array_2[0,i], col.nu_array_2[0,i]] for i in range(np.shape(col.x_array_2)[1])]

                               for col in columns], []))
print('3')
xy_points_with_M = np.array(sum([[[col.x_array[0,i], col.y_array[0,i], col.M_array[0,i]] for i in range(np.shape(col.x_array)[1])]
                               + [[col.x_array_2[0,i], col.y_array_2[0,i], col.M_array_2[0,i]] for i in range(np.shape(col.x_array_2)[1])]

                               for col in columns], []))
print('4')
xy_points_with_P_over_P_e = np.array(sum([[[col.x_array[0,i], col.y_array[0,i], col.P_over_P_e_array[0,i]] for i in range(np.shape(col.x_array)[1])]
                               + [[col.x_array_2[0,i], col.y_array_2[0,i], col.P_over_P_e_array_2[0,i]] for i in range(np.shape(col.x_array_2)[1])]

                               for col in columns], []))
print('5')

# Calculate extra points between first column and nozzle exit using analytical method.
prerunpoints = generate_pre_init_points(radius, phi_edges[0], init_column.x_array, init_column.y_array, n_vert)
print('5a')
extra_phi, extra_nu, extra_M, extra_PP = get_xy_with_vars(prerunpoints, init_column.phi_edge, init_column.nozzle_exit_mach, init_column.nu_edge, radius, gamma)
print('5b')
xy_points_with_phi = np.concatenate((xy_points_with_phi, extra_phi), axis=0)
print('5c')
xy_points_with_nu = np.concatenate((xy_points_with_nu, extra_nu), axis=0)
print('5d')
xy_points_with_M = np.concatenate((xy_points_with_M, extra_M), axis=0)
print('5e')
xy_points_with_P_over_P_e = np.concatenate((xy_points_with_P_over_P_e, extra_PP), axis=0)

# Get the points at the atmospheric boundary for plotting the edge.
top_points =np.array(sum([[[col.x_array[0,-1], col.y_array[0,-1], col.phi_array[0,-1]]]
                               for col in columns], []))

print('9')
# Create interpolating functions for the flow properties.
interp_edge = interp1d(top_points[:,0], top_points[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
print('10a')
inter_right_edge = interp1d(new_column.y_array.ravel(), new_column.x_array.ravel(), kind='linear', bounds_error=False, fill_value='extrapolate')
print('10b')
interp_phi = NearestNDInterpolator(xy_points_with_phi[:,0:2], xy_points_with_phi[:,2])
print('11')
interp_nu = NearestNDInterpolator(xy_points_with_nu[:,0:2], xy_points_with_nu[:,2])
print('12')
interp_M = NearestNDInterpolator(xy_points_with_M[:,0:2], xy_points_with_M[:,2])
print('13')
interp_P_over_P_e = NearestNDInterpolator(xy_points_with_P_over_P_e[:,0:2], xy_points_with_P_over_P_e[:,2])
print('14')
def interp_V_Plus(X,Y): return interp_nu(X,Y) - interp_phi(X,Y)

def interp_V_Minus(X,Y): return interp_nu(X,Y) + interp_phi(X,Y)
print('15')

# Plot V-
X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 2000), np.linspace(0, np.max(xy_points_with_phi[:,1]), 2000))
Z = interp_V_Minus(X, Y)
print('16')
y_edge = interp_edge(X)
y_right_edge = inter_right_edge(Y)
mask = (Y > y_edge) | (X > y_right_edge)
Z_masked = np.ma.array(Z, mask=mask)
print('17')


fig, ax = plt.subplots(dpi=600, figsize = (8,3))
cm = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis')
ax.contour(X, Y, Z_masked, colors='black', linewidths=0.3)
plt.plot(X[0, :], interp_edge(X[0, :]), 'k-', linewidth=2)
plt.plot(new_column.x_array.ravel(), new_column.y_array.ravel(), 'k-', linewidth=2)
plt.plot(init_column.x_array.ravel(), init_column.y_array.ravel(), 'k--', linewidth=2)
ax.set_aspect('equal')
plt.title("V-")
plt.colorbar(cm, location='bottom')

# Plot V+
X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 2000), np.linspace(0, np.max(xy_points_with_phi[:,1]), 2000))
Z = interp_V_Plus(X, Y)
y_edge = interp_edge(X)
y_right_edge = inter_right_edge(Y)
mask = (Y > y_edge) | (X > y_right_edge)
Z_masked = np.ma.array(Z, mask=mask)

fig2, ax2 = plt.subplots(dpi=600, figsize = (8,3))
cm3 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis')
ax2.contour(X, Y, Z_masked, colors='black', linewidths=0.3)
plt.plot(X[0, :], interp_edge(X[0, :]), 'k-', linewidth=2)
plt.plot(new_column.x_array.ravel(), new_column.y_array.ravel(), 'k-', linewidth=2)
plt.plot(init_column.x_array.ravel(), init_column.y_array.ravel(), 'k--', linewidth=2)
ax2.set_aspect('equal')
plt.title("V+")
plt.colorbar(cm3, location='bottom')

# Plot vector field
X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 30), np.linspace(0, np.max(xy_points_with_phi[:,1]), 30))
Z1, Z2 = np.cos(interp_phi(X, Y)), np.sin(interp_phi(X, Y))
y_edge = interp_edge(X)
y_right_edge = inter_right_edge(Y)
mask = (Y > y_edge) | (X > y_right_edge)
Z_masked1, Z_masked2 = np.ma.filled(np.ma.array(Z1, mask=mask), fill_value=np.nan), np.ma.filled(np.ma.array(Z2, mask=mask), fill_value=np.nan)

fig3, ax3 = plt.subplots(dpi=600, figsize=(8,2))
q = ax3.quiver(X, Y, Z_masked1, Z_masked2, headaxislength=0)
plt.plot(X[0, :], interp_edge(X[0, :]), 'k-', linewidth=2)
plt.plot(new_column.x_array.ravel(), new_column.y_array.ravel(), 'k-', linewidth=2)
ax3.set_aspect('equal')
plt.title("Vector Field")

# Plot phi
X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 2000), np.linspace(0, np.max(xy_points_with_phi[:,1]), 2000))
Z = interp_phi(X, Y)
y_edge = interp_edge(X)
y_right_edge = inter_right_edge(Y)
mask = (Y > y_edge) | (X > y_right_edge)
Z_masked = np.ma.array(Z, mask=mask)

fig4, ax4 = plt.subplots(dpi=600, figsize = (8,3))
cm4 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis')
plt.title("Phi Distribution")
plt.plot(X[0, :], interp_edge(X[0, :]), 'k-', linewidth=2)
plt.plot(new_column.x_array.ravel(), new_column.y_array.ravel(), 'k-', linewidth=2)
plt.plot(init_column.x_array.ravel(), init_column.y_array.ravel(), 'k--', linewidth=2)
ax4.set_aspect('equal')
plt.colorbar(cm4, location='bottom')

# Plot P/P_e
X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 2000), np.linspace(0, np.max(xy_points_with_phi[:,1]), 2000))
Z = interp_P_over_P_e(X, Y)
y_edge = interp_edge(X)
y_right_edge = inter_right_edge(Y)
mask = (Y > y_edge) | (X > y_right_edge)
Z_masked = np.ma.array(Z, mask=mask).filled(1.)

fig5, ax5 = plt.subplots(dpi=600, figsize = (8,3))
cm5 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='afmhot', norm=matplotlib.colors.Normalize(vmin=0., vmax=1.))
plt.title("Pressure ratio distribution")
plt.plot(X[0, :], interp_edge(X[0, :]), 'k-', linewidth=2)
plt.plot(new_column.x_array.ravel(), new_column.y_array.ravel(), 'k-', linewidth=2)
plt.plot(init_column.x_array.ravel(), init_column.y_array.ravel(), 'k--', linewidth=2)
ax5.set_aspect('equal')
plt.colorbar(cm5, location='bottom')




# Plot M
X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 2000), np.linspace(0, np.max(xy_points_with_phi[:,1]), 2000))
Z = interp_M(X, Y)
y_edge = interp_edge(X)
y_right_edge = inter_right_edge(Y)
mask = (Y > y_edge) | (X > y_right_edge)
Z_masked = np.ma.array(Z, mask=mask)

fig6, ax6 = plt.subplots(dpi=600, figsize = (8,3))
cm6 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis')
plt.title("Mach number distribution")
plt.plot(X[0, :], interp_edge(X[0, :]), 'k-', linewidth=2)
plt.plot(new_column.x_array.ravel(), new_column.y_array.ravel(), 'k-', linewidth=2)
plt.plot(init_column.x_array.ravel(), init_column.y_array.ravel(), 'k--', linewidth=2)
ax6.set_aspect('equal')
plt.colorbar(cm6, location='bottom')

# Plot nu
X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 2000), np.linspace(0, np.max(xy_points_with_phi[:,1]), 2000))
Z = interp_nu(X, Y)
y_edge = interp_edge(X)
y_right_edge = inter_right_edge(Y)
mask = (Y > y_edge) | (X > y_right_edge)
Z_masked = np.ma.array(Z, mask=mask)

fig7, ax7 = plt.subplots(dpi=600, figsize = (8,3))
cm7 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis')
plt.title("nu distribution")
plt.plot(X[0, :], interp_edge(X[0, :]), 'k-', linewidth=2)
plt.plot(new_column.x_array.ravel(), new_column.y_array.ravel(), 'k-', linewidth=2)
plt.plot(init_column.x_array.ravel(), init_column.y_array.ravel(), 'k--', linewidth=2)
ax7.set_aspect('equal')
plt.colorbar(cm7, location='bottom')



# Plot streamlines
X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 300), np.linspace(0, np.max(xy_points_with_phi[:,1]), 300))
Z1, Z2 = np.cos(interp_phi(X, Y)), np.sin(interp_phi(X, Y))
y_edge = interp_edge(X)
y_right_edge = inter_right_edge(Y)
mask = (Y > y_edge) | (X > y_right_edge)
Z_masked1, Z_masked2 = np.ma.filled(np.ma.array(Z1, mask=mask), fill_value=np.nan), np.ma.filled(np.ma.array(Z2, mask=mask), fill_value=np.nan)

fig8, ax8 = plt.subplots(dpi=600, figsize = (8,2))
ax8.streamplot(X, Y, Z_masked1, Z_masked2, density=1)
plt.plot(X[0, :], interp_edge(X[0, :]), 'k-', linewidth=2)
plt.plot(new_column.x_array.ravel(), new_column.y_array.ravel(), 'k-', linewidth=2)
ax3.set_aspect('equal')
plt.title('Streamlines')

plt.show()
print('final')