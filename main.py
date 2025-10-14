import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import CloughTocher2DInterpolator, interp1d, NearestNDInterpolator
from scipy.spatial import Delaunay


# Define marching column class:

def prandtl_meyer(M, gamma):
    if M<1e6:
        return np.sqrt((gamma+1)/(gamma-1))*np.arctan(np.sqrt(((gamma-1)/(gamma+1))*(M**2-1))) - np.arctan(np.sqrt(M**2-1))
    else:
        return np.inf

def inverse_prandtl_meyer(nu_target, gamma, tol=1e-10, maxiter=60):
    if nu_target < 0:
        return np.nan
    # Fast special-case: zero
    if abs(nu_target) == 0.0:
        return 1.0

    def dnu_dM(M, gamma=1.4):
        if M <= 1.0:
            raise ValueError("M must be greater than 1")
        return 2.0 * np.sqrt(M * M - 1.0) / (M * ((gamma - 1) * M * M + 2.0))
    M_lo = 1.0
    M_hi = 2.0
    while prandtl_meyer(M_hi, gamma) < nu_target:
        M_hi *= 2.0

    nu_lo = 0.0
    nu_hi = prandtl_meyer(M_hi, gamma)
    M = M_lo + (nu_target - nu_lo) * (M_hi - M_lo) / (nu_hi - nu_lo)

    for i in range(maxiter):
        nu_M = prandtl_meyer(M, gamma)
        f = nu_M - nu_target
        if abs(f) < tol:
            return M
        dfdM = dnu_dM(M, gamma)
        if dfdM != 0.0:
            M_new = M - f / dfdM
        else:
            M_new = (M_lo + M_hi) / 2.0
        if (M_new <= M_lo) or (M_new >= M_hi) or (M_new <= 1.0) or not (np.isfinite(M_new)):
            M_new = 0.5 * (M_lo + M_hi)
        if prandtl_meyer(M_new, gamma) > nu_target:
            M_hi = M_new
        else:
            M_lo = M_new

        M = M_new
    return M



class column:
    def __init__(self, n_points):
        self.x_array = np.zeros((1,n_points))
        self.y_array = np.zeros((1,n_points))
        self.phi_array = np.zeros((1,n_points))
        self.nu_array = np.zeros((1,n_points))
        self.M_array = np.zeros((1,n_points))
        self.P_over_P_e_array = np.zeros((1,n_points))
        self.x_array_2 = np.zeros((1, n_points-1))
        self.y_array_2 = np.zeros((1, n_points-1))
        self.phi_array_2 = np.zeros((1, n_points-1))
        self.nu_array_2 = np.zeros((1, n_points-1))
        self.M_array_2 = np.zeros((1, n_points-1))
        self.P_over_P_e_array_2 = np.zeros((1, n_points-1))
        self.n_points = n_points
        self.nozzle_exit_mach = 0
        self.stop = False

    def init_space_march(self, phi_edge, radius, M_e, gamma):
        self.y_array = np.linspace(0, radius, self.n_points).reshape((1,self.n_points))
        self.nu_array = np.ones((1, self.n_points)) * prandtl_meyer(M_e, gamma)
        self.M_array = np.ones((1, self.n_points)) * M_e
        self.P_over_P_e_array = np.ones((1, self.n_points))
        self.nozzle_exit_mach = M_e
        self.phi_array = np.linspace(0, phi_edge, self.n_points).reshape((1,self.n_points))
        for i in range(self.n_points-1):
            phi_A = self.phi_array[0,i]
            phi_B = self.phi_array[0,i+1]
            nu_A = self.nu_array[0,i]
            nu_B = self.nu_array[0,i+1]
            M_a = inverse_prandtl_meyer(nu_A, gamma)
            M_b = inverse_prandtl_meyer(nu_B, gamma)
            mu_a = np.arcsin(1/M_a)
            mu_b = np.arcsin(1/M_b)
            self.phi_array_2[0,i] = 0.5*(phi_A + phi_B + nu_B - nu_A)
            self.nu_array_2[0,i] = 0.5*(nu_A + nu_B + phi_B - phi_A)
            M_p = inverse_prandtl_meyer(self.nu_array_2[0,i], gamma)
            mu_p = np.arcsin(1/M_p)
            a_A = 0.5*(phi_A + self.phi_array_2[0,i] + mu_a + mu_p)
            a_B = 0.5*(phi_B - mu_b + self.phi_array_2[0,i] - mu_p)
            self.x_array_2[0,i] = (self.y_array[0,i+1] - self.y_array[0,i] + self.x_array[0,i]*np.tan(a_A) - self.x_array[0,i+1]*np.tan(a_B)) / (np.tan(a_A) - np.tan(a_B))
            self.y_array_2[0,i] = self.y_array[0,i] + (self.x_array_2[0,i]-self.x_array[0,i])*np.tan(a_A)
            self.M_array_2[0,i] = M_p
            self.P_over_P_e_array_2[0,i] = ((1+0.5*(gamma-1)*self.M_array_2[0,i]**2)/(1+0.5*(gamma-1)*self.nozzle_exit_mach**2))**(-gamma/(gamma+1))

    def next_step(self, gamma, next_phi_edge):
        next_column = column(self.n_points)
        next_column.nozzle_exit_mach = self.nozzle_exit_mach
        next_column.x_array[0,0] = self.x_array_2[0,0] - self.y_array_2[0,0]/np.tan(self.phi_array_2[0,0] - np.arcsin(1/inverse_prandtl_meyer(self.nu_array_2[0,0], gamma)))
        next_column.nu_array[0,0] = self.nu_array_2[0,0] + self.phi_array_2[0,0]
        next_column.M_array[0,0] = inverse_prandtl_meyer(next_column.nu_array[0,0], gamma)
        next_column.P_over_P_e_array[0, 0] = ((1 + 0.5 * (gamma - 1) * next_column.M_array[0, 0] ** 2) / (
                1 + 0.5 * (gamma - 1) * next_column.nozzle_exit_mach ** 2)) ** (-gamma / (gamma + 1))
        for i in range(1, self.n_points-1):
            phi_A = self.phi_array_2[0, i-1]
            phi_B = self.phi_array_2[0, i]
            nu_A = self.nu_array_2[0, i-1]
            nu_B = self.nu_array_2[0, i]
            M_a = inverse_prandtl_meyer(nu_A, gamma)
            M_b = inverse_prandtl_meyer(nu_B, gamma)
            mu_a = np.arcsin(1 / M_a)
            mu_b = np.arcsin(1 / M_b)
            next_column.phi_array[0, i] = 0.5 * (phi_A + phi_B + nu_B - nu_A)
            next_column.nu_array[0, i] = 0.5 * (nu_A + nu_B + phi_B - phi_A)
            M_p = inverse_prandtl_meyer(next_column.nu_array[0, i], gamma)
            next_column.M_array[0, i] = M_p
            next_column.P_over_P_e_array[0,i] = ((1+0.5*(gamma-1)*next_column.M_array[0,i]**2)/(1+0.5*(gamma-1)*next_column.nozzle_exit_mach**2))**(-gamma/(gamma+1))
            mu_p = np.arcsin(1 / M_p)
            a_A = 0.5 * (phi_A + next_column.phi_array[0, i] + mu_a + mu_p)
            a_B = 0.5 * (phi_B - mu_b + next_column.phi_array[0, i] - mu_p)
            next_column.x_array[0, i] = (self.y_array_2[0, i] - self.y_array_2[0, i-1] + self.x_array_2[0,i-1]*np.tan(a_A) - self.x_array_2[0,i]*np.tan(a_B)) / (np.tan(a_A) - np.tan(a_B))
            next_column.y_array[0, i] = self.y_array_2[0, i-1] + (next_column.x_array[0, i] - self.x_array_2[0,i-1]) * np.tan(a_A)
        next_column.phi_array[0, -1] = next_phi_edge
        next_column.nu_array[0, -1] = next_column.phi_array[0, -1] + self.nu_array_2[0, -1] - self.phi_array_2[0, -1]
        next_column.M_array[0, -1] = inverse_prandtl_meyer(next_column.nu_array[0, -1], gamma)
        next_column.P_over_P_e_array[0, -1] = ((1 + 0.5 * (gamma - 1) * next_column.M_array[0, -1] ** 2) / (
                    1 + 0.5 * (gamma - 1) * next_column.nozzle_exit_mach ** 2)) ** (-gamma / (gamma + 1))

        Alp = 0.5*(self.phi_array_2[0,-1] + np.arcsin(1/inverse_prandtl_meyer(self.nu_array_2[0,-1], gamma)) + next_column.phi_array[0,-1] + np.arcsin(1/inverse_prandtl_meyer(next_column.nu_array[0,-1], gamma)))
        Dy = (np.tan(Alp)*(next_column.x_array[0, -2] - self.x_array_2[0, -1]) + self.y_array_2[0, -1] - next_column.y_array[0, -2]) / (1 + np.tan(Alp) * np.tan(next_column.phi_array[0, -1]))
        Dx = Dy * np.tan(next_column.phi_array[0, -1])
        next_column.x_array[0, -1] = next_column.x_array[0, -2] - Dx
        next_column.y_array[0, -1] = next_column.y_array[0, -2] + Dy
        for i in range(self.n_points-1):
            phi_A = next_column.phi_array[0,i]
            phi_B = next_column.phi_array[0,i+1]
            nu_A = next_column.nu_array[0,i]
            nu_B = next_column.nu_array[0,i+1]
            M_a = inverse_prandtl_meyer(nu_A, gamma)
            M_b = inverse_prandtl_meyer(nu_B, gamma)
            mu_a = np.arcsin(1/M_a)
            mu_b = np.arcsin(1/M_b)
            next_column.phi_array_2[0,i] = 0.5*(phi_A + phi_B + nu_B - nu_A)
            next_column.nu_array_2[0,i] = 0.5*(nu_A + nu_B + phi_B - phi_A)
            M_p = inverse_prandtl_meyer(next_column.nu_array_2[0,i], gamma)
            next_column.M_array_2[0,i] = M_p
            next_column.P_over_P_e_array_2[0,i] = ((1+0.5*(gamma-1)*next_column.M_array_2[0,i]**2)/(1+0.5*(gamma-1)*next_column.nozzle_exit_mach**2))**(-gamma/(gamma+1))
            mu_p = np.arcsin(1/M_p)
            a_A = 0.5*(phi_A + next_column.phi_array_2[0,i] + mu_a + mu_p)
            a_B = 0.5*(phi_B - mu_b + next_column.phi_array_2[0,i] - mu_p)
            next_column.x_array_2[0,i] = (next_column.y_array[0,i+1] - next_column.y_array[0,i] + next_column.x_array[0,i]*np.tan(a_A) - next_column.x_array[0,i+1]*np.tan(a_B)) / (np.tan(a_A) - np.tan(a_B))
            next_column.y_array_2[0,i] = next_column.y_array[0,i] + (next_column.x_array_2[0,i] - next_column.x_array[0,i])*np.tan(a_A)
            if next_column.x_array_2[0,i] < next_column.x_array[0,i]:
                next_column.stop = True



        return next_column


phi_edges = np.concatenate((np.zeros(50), np.linspace(0, np.pi/36, 50), np.linspace(np.pi/36, 0, 50), np.zeros(120)))
columns = []
n_vert = 50
init_column = column(n_vert)
init_column.init_space_march(phi_edges[0], 3, 1.7, 1.4)
columns.append(init_column)
for phi_edge in phi_edges[1:-1]:
    new_column = columns[-1].next_step(1.4, phi_edge)
    if new_column.stop == False:
        columns.append(new_column)

xy_points_with_phi = np.array(sum([[[col.x_array[0,i], col.y_array[0,i], col.phi_array[0,i]] for i in range(np.shape(col.x_array)[1])]
                               + [[col.x_array_2[0,i], col.y_array_2[0,i], col.phi_array_2[0,i]] for i in range(np.shape(col.x_array_2)[1])]

                               for col in columns], []))
xy_points_with_nu = np.array(sum([[[col.x_array[0,i], col.y_array[0,i], col.nu_array[0,i]] for i in range(np.shape(col.x_array)[1])]
                               + [[col.x_array_2[0,i], col.y_array_2[0,i], col.nu_array_2[0,i]] for i in range(np.shape(col.x_array_2)[1])]

                               for col in columns], []))
xy_points_with_M = np.array(sum([[[col.x_array[0,i], col.y_array[0,i], col.M_array[0,i]] for i in range(np.shape(col.x_array)[1])]
                               + [[col.x_array_2[0,i], col.y_array_2[0,i], col.M_array_2[0,i]] for i in range(np.shape(col.x_array_2)[1])]

                               for col in columns], []))
xy_points_with_P_over_P_e = np.array(sum([[[col.x_array[0,i], col.y_array[0,i], col.P_over_P_e_array[0,i]] for i in range(np.shape(col.x_array)[1])]
                               + [[col.x_array_2[0,i], col.y_array_2[0,i], col.P_over_P_e_array_2[0,i]] for i in range(np.shape(col.x_array_2)[1])]

                               for col in columns], []))

top_points = np.concatenate((np.array(sum([[[col.x_array[0,-1], col.y_array[0,-1], col.phi_array[0,-1]]]


                               for col in columns], [])), xy_points_with_phi[:, :][-n_vert:]))
top_points_2 = np.array(sum([[[col.x_array_2[0,-1], col.y_array_2[0,-1], col.phi_array_2[0,-1]]]

                               for col in columns], []))
interp_edge = interp1d(top_points[:,0], top_points[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
interp_phi = NearestNDInterpolator(xy_points_with_phi[:,0:2], xy_points_with_phi[:,2])
interp_nu = NearestNDInterpolator(xy_points_with_nu[:,0:2], xy_points_with_nu[:,2])
interp_M = NearestNDInterpolator(xy_points_with_M[:,0:2], xy_points_with_M[:,2])
interp_P_over_P_e = NearestNDInterpolator(xy_points_with_P_over_P_e[:,0:2], xy_points_with_P_over_P_e[:,2])

X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 100), np.linspace(0, np.max(xy_points_with_phi[:,1]), 100))
Z = interp_P_over_P_e(X, Y)
y_edge = interp_edge(X)
mask = Y > y_edge
Z_masked = np.ma.array(Z, mask=mask)


plt.figure(dpi=600)
#cm = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='bwr', norm=matplotlib.colors.CenteredNorm(vcenter=0))
cm = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis')
#plt.scatter(top_points[:,0], top_points[:,1], c='black', edgecolor='k', cmap='viridis')
#plt.scatter(top_points_2[:,0], top_points_2[:,1], c='grey', edgecolor='k', cmap='viridis')
cm2 = plt.scatter(xy_points_with_phi[:, 0], xy_points_with_phi[:, 1] ,s=0.5, edgecolor='none')
plt.title("Clough-Tocher (cubic) interpolation of scattered data")
plt.colorbar(cm)
plt.colorbar(cm2)
plt.show()