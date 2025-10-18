import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
from scipy.interpolate import CloughTocher2DInterpolator, interp1d, NearestNDInterpolator
from scipy.spatial import Delaunay

from Generate_pre_init_points import get_xy_with_vars, generate_pre_init_points
from numba_kernels import (
    prandtl_meyer_nb,
    inverse_prandtl_meyer_nb,
    inverse_expansion_fan_function_nb,
    next_step_core_nb,
)
import copy
import Generate_pre_init_points

# Define marching column class:

def prandtl_meyer(M, gamma):
    # Use JIT-optimized kernel
    return prandtl_meyer_nb(M, gamma)

def inverse_prandtl_meyer(nu_target, gamma, tol=1e-10, maxiter=60):
    # Use JIT-optimized kernel
    return inverse_prandtl_meyer_nb(nu_target, gamma, tol, maxiter)

def inverse_expansion_fan_function(psi_target, gamma, mach_nozzle, tol=1e-10, maxiter=60):
    # Use JIT-optimized kernel
    return inverse_expansion_fan_function_nb(psi_target, gamma, mach_nozzle, tol, maxiter)



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

    def next_step(self, gamma, next_phi_edge, custom_nu_edge=0):
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

        stop = next_step_core_nb(
            self.n_points,
            gamma,
            next_phi_edge,
            float(custom_nu_edge),
            next_column.nozzle_exit_mach,
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
        next_column.stop = bool(stop)
        return next_column

    def backward_step(self, gamma, prev_phi_edge, radius=3):
        prev_column = column(self.n_points)
        prev_column.nozzle_exit_mach = self.nozzle_exit_mach
        for i in range(self.n_points-1):
            phi_A = self.phi_array[0, i]
            phi_B = self.phi_array[0, i+1]
            nu_A = self.nu_array[0, i]
            nu_B = self.nu_array[0, i+1]
            M_a = inverse_prandtl_meyer(nu_A, gamma)
            M_b = inverse_prandtl_meyer(nu_B, gamma)
            mu_a = np.arcsin(1 / M_a)
            mu_b = np.arcsin(1 / M_b)
            prev_column.phi_array_2[0, i] = 0.5 * (phi_A + phi_B + nu_A - nu_B)
            prev_column.nu_array_2[0, i] = 0.5 * (nu_A + nu_B + phi_A - phi_B)
            M_p = inverse_prandtl_meyer(prev_column.nu_array_2[0, i], gamma)
            prev_column.M_array_2[0, i] = M_p
            prev_column.P_over_P_e_array_2[0,i] = ((1+0.5*(gamma-1)*prev_column.M_array_2[0,i]**2)/(1+0.5*(gamma-1)*prev_column.nozzle_exit_mach**2))**(-gamma/(gamma+1))
            mu_p = np.arcsin(1 / M_p)
            a_B = 0.5 * (phi_B + prev_column.phi_array_2[0, i] + mu_b + mu_p)
            a_A = 0.5 * (phi_A - mu_a + prev_column.phi_array_2[0, i] - mu_p)
            prev_column.x_array_2[0, i] = (self.y_array[0, i+1] - self.y_array[0, i] - self.x_array[0,i+1]*np.tan(a_B) + self.x_array[0,i]*np.tan(a_A)) / (np.tan(a_A) - np.tan(a_B))
            prev_column.y_array_2[0, i] = self.y_array[0, i] + (prev_column.x_array_2[0, i] - self.x_array[0,i]) * np.tan(a_A)

            if prev_column.y_array_2[0, i] < radius - 1/(np.sqrt(self.nozzle_exit_mach**2 - 1)) * prev_column.x_array_2[0, i]:
                prev_column.phi_array_2[0, 1] = 0
                prev_column.nu_array_2[0, 1] = prandtl_meyer(prev_column.nozzle_exit_mach, gamma)
                prev_column.M_array_2[0, 1] = prev_column.nozzle_exit_mach
                prev_column.P_over_P_e_array_2[0, 1] = ((1 + 0.5 * (gamma - 1) * prev_column.M_array_2[0, 1] ** 2) / (
                            1 + 0.5 * (gamma - 1) * prev_column.nozzle_exit_mach ** 2)) ** (-gamma / (gamma + 1))
                mu_p = np.arcsin(1 / M_p)


            if (prev_column.phi_array_2[0, i] + mu_p > np.pi/2) or ((prev_column.phi_array_2[0, i] - mu_p < -np.pi/2)):
                prev_column.stop = True
        prev_column.phi_array[0, 0] = 0
        prev_column.nu_array[0, 0] = prandtl_meyer(prev_column.nozzle_exit_mach, gamma)
        prev_column.M_array[0, 0] = prev_column.nozzle_exit_mach
        prev_column.P_over_P_e_array[0, 0] = ((1 + 0.5 * (gamma - 1) * prev_column.M_array[0, 0] ** 2) / (
                    1 + 0.5 * (gamma - 1) * prev_column.nozzle_exit_mach ** 2)) ** (-gamma / (gamma + 1))
        prev_column.x_array[0, 0] = prev_column.x_array_2[0,0] - prev_column.y_array_2[0,0] / (np.tan(prev_column.phi_array_2[0,0] + np.arcsin(1/(inverse_prandtl_meyer(prev_column.nu_array_2[0,0], gamma)))))
        prev_column.y_array[0, 0] = 0
        prev_column.phi_array[0, -1] = prev_phi_edge
        prev_column.nu_array[0, -1] = prev_column.phi_array[0, -1] + prandtl_meyer(prev_column.nozzle_exit_mach, gamma)
        prev_column.M_array[0, -1] = inverse_prandtl_meyer(prev_column.nu_array[0, -1], gamma)
        prev_column.P_over_P_e_array[0, -1] = ((1 + 0.5 * (gamma - 1) * prev_column.M_array[0, -1] ** 2) / (
                    1 + 0.5 * (gamma - 1) * prev_column.nozzle_exit_mach ** 2)) ** (-gamma / (gamma + 1))

        for i in range(1, self.n_points-1):
            phi_A = prev_column.phi_array_2[0, i-1]
            phi_B = prev_column.phi_array_2[0, i]
            nu_A = prev_column.nu_array_2[0, i-1]
            nu_B = prev_column.nu_array_2[0, i]
            M_a = inverse_prandtl_meyer(nu_A, gamma)
            M_b = inverse_prandtl_meyer(nu_B, gamma)
            mu_a = np.arcsin(1 / M_a)
            mu_b = np.arcsin(1 / M_b)
            prev_column.phi_array[0, i] = 0.5 * (phi_A + phi_B + nu_A - nu_B)
            prev_column.nu_array[0, i] = 0.5 * (nu_A + nu_B + phi_A - phi_B)
            M_p = inverse_prandtl_meyer(prev_column.nu_array[0, i], gamma)
            prev_column.M_array[0, i] = M_p
            prev_column.P_over_P_e_array[0,i] = ((1+0.5*(gamma-1)*prev_column.M_array[0,i]**2)/(1+0.5*(gamma-1)*prev_column.nozzle_exit_mach**2))**(-gamma/(gamma+1))
            mu_p = np.arcsin(1 / M_p)
            a_B = 0.5 * (phi_B + prev_column.phi_array[0, i] + mu_b + mu_p)
            a_A = 0.5 * (phi_A - mu_a + prev_column.phi_array[0, i] - mu_p)
            prev_column.x_array[0, i] = (prev_column.y_array_2[0, i] - prev_column.y_array_2[0, i-1] - prev_column.x_array_2[0,i]*np.tan(a_B) + prev_column.x_array_2[0,i-1]*np.tan(a_A)) / (np.tan(a_A) - np.tan(a_B))
            prev_column.y_array[0, i] = prev_column.y_array_2[0, i-1] + (prev_column.x_array[0, i-1] - prev_column.x_array_2[0,i-1]) * np.tan(a_A)

            if prev_column.y_array[0, i] < radius - 1 / (np.sqrt(self.nozzle_exit_mach ** 2 - 1)) * \
                    prev_column.x_array[0, i]:
                prev_column.phi_array[0, 1] = 0
                prev_column.nu_array[0, 1] = prandtl_meyer(prev_column.nozzle_exit_mach, gamma)
                prev_column.M_array[0, 1] = prev_column.nozzle_exit_mach
                prev_column.P_over_P_e_array[0, 1] = ((1 + 0.5 * (gamma - 1) * prev_column.M_array[0, 1] ** 2) / (
                        1 + 0.5 * (gamma - 1) * prev_column.nozzle_exit_mach ** 2)) ** (-gamma / (gamma + 1))
                mu_p = np.arcsin(1 / M_p)

            #if prev_column.x_array[0,i] > prev_column.x_array_2[0,i]:
            if (prev_column.phi_array[0, i] + mu_p > np.pi/2) or ((prev_column.phi_array[0, i] - mu_p < -np.pi/2)):
                prev_column.stop = True
                print('stopping because of spacelike')
            if np.max(prev_column.x_array) - np.min(prev_column.x_array) > 20:
                prev_column.stop = True
                print('stopping because of excessive length')

        # Alp = 0.5 * (self.phi_array[0, -1] - np.arcsin(1 / inverse_prandtl_meyer(self.nu_array[0, -1], gamma)) +
        #              prev_column.phi_array_2[0, -1] - np.arcsin(
        #             1 / inverse_prandtl_meyer(prev_column.nu_array_2[0, -1], gamma)))
        # Dy = (np.tan(Alp) * (-prev_column.x_array[0, -2] + prev_column.x_array_2[0, -1]) + prev_column.y_array_2[
        #     0, -1] - prev_column.y_array[0, -2]) / (1 + np.tan(Alp) * np.tan(-prev_column.phi_array[0, -1]))
        # Dx = Dy * np.tan(-prev_column.phi_array[0, -1])
        # prev_column.x_array[0, -1] = prev_column.x_array[0, -2] + Dx
        # prev_column.y_array[0, -1] = prev_column.y_array[0, -2] + Dy
        #Dy = np.sin(Alp)
        R = -1/np.tan(prev_column.phi_array[0,-1]) #(prev_column.y_array[0, -2] - prev_column.y_array[0, -3]) / (
         #           prev_column.x_array[0, -2] - prev_column.x_array[0, -3])
        prev_column.x_array[0, -1] = (1 / (R - np.tan(prev_column.phi_array[0, -1]))) * (
                    self.y_array[0, -1] - self.x_array[0, -1] * np.tan(prev_column.phi_array[0, -1]) -
                    prev_column.y_array[0, -2] + prev_column.x_array[0, -2] * R)
        prev_column.y_array[0, -1] = self.y_array[0, -1] + np.tan(prev_column.phi_array[0, -1]) * (
                    prev_column.x_array[0, -1] - self.x_array[0, -1])
        # Ds = np.sqrt((self.x_array[0,-2] - prev_column.x_array[0,-2])**2 + (self.y_array[0,-2] - prev_column.y_array[0,-2])**2)
        # Dy = Ds * np.sin(self.phi_array[0,-1])
        # Dx =  - Ds * np.cos(self.phi_array[0,-1])
        # prev_column.x_array[0, -1] = self.x_array[0,-1] + Dx
        # prev_column.y_array[0, -1] = self.y_array[0,-1] + Dy

        return prev_column


#phi_edges = np.concatenate((np.zeros(50), np.linspace(0, np.pi/36, 50), np.linspace(np.pi/36, 0, 50), np.zeros(120)))
columns = []
n_vert = 1000
init_column = column(n_vert)
radius = 3
init_column.init_space_march(0, radius, 2, 1.4)
columns.append(init_column)
P_atmos_to_P_e = 0.85
counter=0
phi_edges = []
# counter += 1
custom_nu_edge = prandtl_meyer(np.sqrt((2/(1.4-1)) * (1 + 0.5*(1.4-1)*columns[-1].nozzle_exit_mach**2)*(P_atmos_to_P_e)**((-1.4-1)/1.4)-2/(1.4-1)), 1.4)
phi_edge = custom_nu_edge - prandtl_meyer(init_column.nozzle_exit_mach, 1.4)
# phi_edges.append(phi_edge)
# print(counter, phi_edge)
# new_column = columns[-1].next_step(1.4, phi_edge, custom_nu_edge=custom_nu_edge)
mu_1 = np.arcsin(1/columns[-1].nozzle_exit_mach)
mu_2 = np.arcsin(1/inverse_prandtl_meyer(custom_nu_edge, 1.4))
init_column.x_array[0,0] = radius/np.tan(mu_1)
for i in range(1, n_vert):
    if i <= np.ceil((1-mu_2/(mu_1+phi_edge))*n_vert):
        Dpsi = (phi_edge - mu_2 + mu_1)/np.ceil((1-mu_2/(mu_1+phi_edge))*n_vert)
        #prev_psi = np.arctan((init_column.y_array[0, i-1] - radius) / (init_column.x_array[0, i-1] - 0))
        psi = -mu_1 + i*(phi_edge-mu_2 + mu_1)/np.ceil((1-mu_2/(mu_1+phi_edge))*n_vert)
        #psi = prev_psi + Dpsi
        prev_psi = psi - (phi_edge-mu_2 + mu_1)/np.ceil((1-mu_2/(mu_1+phi_edge))*n_vert)
        Ds = np.cos(psi - np.arctan((init_column.x_array[0, i-1] - 0) / (radius - init_column.y_array[0, i-1]))) * np.sqrt((init_column.x_array[0, i-1] - 0)**2 + (radius - init_column.y_array[0, i-1])**2) / np.cos(inverse_expansion_fan_function(prev_psi, 1.4, init_column.nozzle_exit_mach) - psi)
        phi_local = inverse_expansion_fan_function(prev_psi, 1.4, init_column.nozzle_exit_mach)
        Dy = Ds * np.cos(phi_local)
        Dx = -Dy * np.tan(phi_local)
        init_column.x_array[0, i] = init_column.x_array[0, i-1] + Dx
        init_column.y_array[0, i] = init_column.y_array[0, i-1] + Dy
        init_column.phi_array[0, i] = inverse_expansion_fan_function(psi, 1.4, init_column.nozzle_exit_mach)
        init_column.nu_array[0, i] = prandtl_meyer(init_column.nozzle_exit_mach, 1.4) + init_column.phi_array[0, i]
        init_column.M_array[0, i] = inverse_prandtl_meyer(init_column.nu_array[0, i], 1.4)
        init_column.P_over_P_e_array[0, i] = ((1+0.5*(1.4-1)*init_column.M_array_2[0,i]**2)/(1+0.5*(1.4-1)*init_column.nozzle_exit_mach**2))**(-1.4/(1.4+1))
    else:
        Dpsi = (mu_2)/(n_vert - np.ceil((1-mu_2/(mu_1+phi_edge))*n_vert) - 1)
        #prev_psi = np.arctan((init_column.y_array[0, i-1] - radius) / (init_column.x_array[0, i-1] - 0))
        psi = phi_edge - mu_2 + (i-np.ceil((1-mu_2/(mu_1+phi_edge))*n_vert))*Dpsi
        prev_psi = psi - Dpsi
        #psi = prev_psi + Dpsi
        if psi <= 0.:
            Ds = np.cos(psi - np.arctan((init_column.x_array[0, i-1] - 0) / (radius - init_column.y_array[0, i-1]))) * np.sqrt((init_column.x_array[0, i-1] - 0)**2 + (radius - init_column.y_array[0, i-1])**2) / np.cos(phi_edge - psi)
        else:
            Ds = np.sin(psi - np.arctan((init_column.y_array[0, i-1] - radius) / (init_column.x_array[0, i-1] - 0))) * np.sqrt((init_column.x_array[0, i-1] - 0)**2 + (init_column.y_array[0, i-1] - radius)**2) / np.cos(-phi_edge + np.arctan((init_column.y_array[0, i-1] - radius) / (init_column.x_array[0, i-1] - 0)))
        Dy = Ds * np.cos(phi_edge)
        Dx = -Dy * np.tan(phi_edge)
        init_column.x_array[0, i] = init_column.x_array[0, i-1] + Dx
        init_column.y_array[0, i] = init_column.y_array[0, i-1] + Dy
        init_column.phi_array[0, i] = phi_edge
        init_column.nu_array[0, i] = prandtl_meyer(init_column.nozzle_exit_mach, 1.4) + phi_edge
        init_column.M_array[0, i] = inverse_prandtl_meyer(init_column.nu_array[0, i], 1.4)
        init_column.P_over_P_e_array[0, i] = ((1+0.5*(1.4-1)*init_column.M_array[0,i]**2)/(1+0.5*(1.4-1)*init_column.nozzle_exit_mach**2))**(-1.4/(1.4+1))
for i in range(init_column.n_points - 1):
    phi_A = init_column.phi_array[0, i]
    phi_B = init_column.phi_array[0, i + 1]
    nu_A = init_column.nu_array[0, i]
    nu_B = init_column.nu_array[0, i + 1]
    M_a = inverse_prandtl_meyer(nu_A, 1.4)
    M_b = inverse_prandtl_meyer(nu_B, 1.4)
    mu_a = np.arcsin(1 / M_a)
    mu_b = np.arcsin(1 / M_b)
    init_column.phi_array_2[0, i] = 0.5 * (phi_A + phi_B + nu_B - nu_A)
    init_column.nu_array_2[0, i] = 0.5 * (nu_A + nu_B + phi_B - phi_A)
    M_p = inverse_prandtl_meyer(init_column.nu_array_2[0, i], 1.4)
    mu_p = np.arcsin(1 / M_p)
    a_A = 0.5 * (phi_A + init_column.phi_array_2[0, i] + mu_a + mu_p)
    a_B = 0.5 * (phi_B - mu_b + init_column.phi_array_2[0, i] - mu_p)
    init_column.x_array_2[0, i] = (init_column.y_array[0, i + 1] - init_column.y_array[0, i] + init_column.x_array[0, i] * np.tan(a_A) -
                            init_column.x_array[0, i + 1] * np.tan(a_B)) / (np.tan(a_A) - np.tan(a_B))
    init_column.y_array_2[0, i] = init_column.y_array[0, i] + (init_column.x_array_2[0, i] - init_column.x_array[0, i]) * np.tan(a_A)
    init_column.M_array_2[0, i] = M_p
    init_column.P_over_P_e_array_2[0, i] = ((1 + 0.5 * (1.4 - 1) * init_column.M_array_2[0, i] ** 2) / (
                1 + 0.5 * (1.4 - 1) * init_column.nozzle_exit_mach ** 2)) ** (-1.4 / (1.4 + 1))

columns[0] = init_column
counter2 = 0
xy = np.concatenate((init_column.x_array_2.T, init_column.y_array_2.T), axis=1)
xy = np.concatenate((xy, np.concatenate((init_column.x_array.T, init_column.y_array.T), axis=1)), axis=0)
xy_init = copy.deepcopy(xy)

stopstop2 = False
while stopstop2 == False and counter2 < 0*100:
    counter2 += 1
    phi_edge = columns[0].phi_array[0,-1] - columns[0].nu_array[0,-1] + prandtl_meyer(np.sqrt((2/(1.4-1)) * (1 + 0.5*(1.4-1)*columns[0].nozzle_exit_mach**2)*(P_atmos_to_P_e)**((-1.4-1)/1.4)-2/(1.4-1)), 1.4)
    print(counter2, phi_edge, stopstop2, columns[0].stop)
    new_column = columns[0].backward_step(1.4, phi_edge, radius)
    phi_edges.insert(0, phi_edge)
    xy = np.concatenate((xy, np.concatenate((new_column.x_array_2.T, new_column.y_array_2.T), axis=1)), axis=0)
    xy = np.concatenate((xy, np.concatenate((new_column.x_array.T, new_column.y_array.T), axis=1)), axis=0)
    if counter2 % 5 == 0:
        xy_init = np.concatenate((xy_init, np.concatenate((new_column.x_array.T, new_column.y_array.T), axis=1)), axis=0)

    if new_column.stop == False:
        columns.insert(0, new_column)
    else:
        stopstop2 = True

stopstop = False
while stopstop == False and counter < 10000:
    counter += 1
    phi_edge = columns[-1].phi_array_2[0,-1] - columns[-1].nu_array_2[0,-1] + prandtl_meyer(np.sqrt((2/(1.4-1)) * (1 + 0.5*(1.4-1)*columns[-1].nozzle_exit_mach**2)*(P_atmos_to_P_e)**((-1.4-1)/1.4)-2/(1.4-1)), 1.4)
    phi_edges.append(phi_edge)
    print(counter, phi_edge, columns[-1].stop, stopstop)
    new_column = columns[-1].next_step(1.4, phi_edge)
    xy = np.concatenate((xy, np.concatenate((new_column.x_array_2.T, new_column.y_array_2.T), axis=1)), axis=0)
    xy = np.concatenate((xy, np.concatenate((new_column.x_array.T, new_column.y_array.T), axis=1)), axis=0)
    if counter % 50 == 0:
        xy_init = np.concatenate((xy_init, np.concatenate((new_column.x_array.T, new_column.y_array.T), axis=1)), axis=0)

    if new_column.stop == False:
        columns.append(new_column)
    else:
        stopstop = True
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
prerunpoints = generate_pre_init_points(radius, phi_edges[0], init_column.x_array, init_column.y_array, n_vert)
print('5a')
extra_phi, extra_nu, extra_M, extra_PP = get_xy_with_vars(prerunpoints, phi_edges[0], init_column.nozzle_exit_mach, custom_nu_edge, radius, 1.4)
print('5b')
xy_points_with_phi = np.concatenate((xy_points_with_phi, extra_phi), axis=0)
print('5c')
xy_points_with_nu = np.concatenate((xy_points_with_nu, extra_nu), axis=0)
print('5d')
xy_points_with_M = np.concatenate((xy_points_with_M, extra_M), axis=0)
print('5e')
xy_points_with_P_over_P_e = np.concatenate((xy_points_with_P_over_P_e, extra_PP), axis=0)
top_points = np.concatenate((np.array(sum([[[col.x_array[0,-1], col.y_array[0,-1], col.phi_array[0,-1]]]


                               for col in columns], [])), xy_points_with_phi[:, :][-n_vert:]))
# print('6')
# top_points = np.concatenate((top_points, np.array([[top_points[0,0], 0, 0]])))
# print('7')
# top_points = np.concatenate((top_points, np.array([[top_points[0,0] - 0.01, 0, 0]])))
# print('8')
top_points_2 = np.array(sum([[[col.x_array_2[0,-1], col.y_array_2[0,-1], col.phi_array_2[0,-1]]]

                               for col in columns], []))
print('9')
interp_edge = interp1d(top_points[:,0], top_points[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
print('10')
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

X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 300), np.linspace(0, np.max(xy_points_with_phi[:,1]), 300))
Z = interp_V_Minus(X, Y)
print('16')
y_edge = interp_edge(X)
mask = Y > y_edge
Z_masked = np.ma.array(Z, mask=mask)
print('17')

# plt.figure()
# plt.plot(phi_edges)

fig, ax = plt.subplots(dpi=600, figsize = (8,6))
#cm = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='bwr', norm=matplotlib.colors.CenteredNorm(vcenter=0))
cm = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis')
ax.contour(X, Y, Z_masked)
#plt.scatter(top_points[:,0], top_points[:,1], c='black', edgecolor='k', cmap='viridis')
#plt.scatter(top_points_2[:,0], top_points_2[:,1], c='grey', edgecolor='k', cmap='viridis')
#cm2 = plt.scatter(xy_points_with_phi[:, 0], xy_points_with_phi[:, 1] ,s=0.5, edgecolor='none')
plt.title("V-")
plt.colorbar(cm)
#plt.colorbar(cm2)

X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 300), np.linspace(0, np.max(xy_points_with_phi[:,1]), 300))
Z = interp_V_Plus(X, Y)
y_edge = interp_edge(X)
mask = Y > y_edge
Z_masked = np.ma.array(Z, mask=mask)

fig2, ax2 = plt.subplots(dpi=600, figsize = (8,6))
#cm = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='bwr', norm=matplotlib.colors.CenteredNorm(vcenter=0))
cm3 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis')
ax2.contour(X, Y, Z_masked)
#plt.scatter(top_points[:,0], top_points[:,1], c='black', edgecolor='k', cmap='viridis')
#plt.scatter(top_points_2[:,0], top_points_2[:,1], c='grey', edgecolor='k', cmap='viridis')
#cm2 = plt.scatter(xy_points_with_phi[:, 0], xy_points_with_phi[:, 1] ,s=0.5, edgecolor='none')
plt.title("V+")
plt.colorbar(cm3)
#plt.colorbar(cm2)

X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 30), np.linspace(0, np.max(xy_points_with_phi[:,1]), 30))
Z1, Z2 = np.cos(interp_phi(X, Y)), np.sin(interp_phi(X, Y))
y_edge = interp_edge(X)
mask = Y > y_edge
Z_masked1, Z_masked2 = np.ma.filled(np.ma.array(Z1, mask=mask), fill_value=np.nan), np.ma.filled(np.ma.array(Z2, mask=mask), fill_value=np.nan)


fig3, ax3 = plt.subplots()
q = ax3.quiver(X, Y, Z_masked1, Z_masked2, headaxislength=0)


X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 300), np.linspace(0, np.max(xy_points_with_phi[:,1]), 300))
Z = interp_phi(X, Y)
y_edge = interp_edge(X)
mask = Y > y_edge
Z_masked = np.ma.array(Z, mask=mask)

fig4, ax4 = plt.subplots(dpi=600, figsize = (8,6))
#cm = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='bwr', norm=matplotlib.colors.CenteredNorm(vcenter=0))
cm4 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis')
#ax4.contour(X, Y, Z_masked)
plt.scatter(top_points[:,0], top_points[:,1], c='black', edgecolor='k', cmap='viridis')
plt.scatter(xy_init[:,0], xy_init[:,1], c='red', s=0.1)
#plt.scatter(top_points_2[:,0], top_points_2[:,1], c='grey', edgecolor='k', cmap='viridis')
#cm2 = plt.scatter(xy_points_with_phi[:, 0], xy_points_with_phi[:, 1] ,s=0.1, edgecolor='none')
plt.title("Phi Distribution")
plt.colorbar(cm4)
#plt.colorbar(cm2)

X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 300), np.linspace(0, np.max(xy_points_with_phi[:,1]), 300))
Z = interp_P_over_P_e(X, Y)
y_edge = interp_edge(X)
mask = Y > y_edge
Z_masked = np.ma.array(Z, mask=mask)

fig5, ax5 = plt.subplots(dpi=600, figsize = (8,6))
cm5 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='bwr', norm=matplotlib.colors.CenteredNorm(vcenter=1))
#cm5 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis')
#ax5.contour(X, Y, Z_masked)
plt.scatter(top_points[:,0], top_points[:,1], c='black', edgecolor='k', cmap='viridis')
#plt.scatter(top_points_2[:,0], top_points_2[:,1], c='grey', edgecolor='k', cmap='viridis')
#cm2 = plt.scatter(xy_points_with_phi[:, 0], xy_points_with_phi[:, 1] ,s=0.1, edgecolor='none')
plt.title("Pressure ratio distribution")
plt.colorbar(cm5)
#plt.colorbar(cm2)


X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 300), np.linspace(0, np.max(xy_points_with_phi[:,1]), 300))
Z = interp_M(X, Y)
y_edge = interp_edge(X)
mask = Y > y_edge
Z_masked = np.ma.array(Z, mask=mask)

fig6, ax6 = plt.subplots(dpi=600, figsize = (8,6))
#cm6 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='bwr', norm=matplotlib.colors.CenteredNorm(vcenter=0))
cm6 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis')
#ax5.contour(X, Y, Z_masked)
plt.scatter(top_points[:,0], top_points[:,1], c='black', edgecolor='k', cmap='viridis')
#plt.scatter(top_points_2[:,0], top_points_2[:,1], c='grey', edgecolor='k', cmap='viridis')
#cm2 = plt.scatter(xy_points_with_phi[:, 0], xy_points_with_phi[:, 1] ,s=0.1, edgecolor='none')
plt.title("Mach number distribution")
#plt.colorbar(cm)
plt.colorbar(cm6)

X,Y = np.meshgrid(np.linspace(0, np.max(xy_points_with_phi[:,0]), 300), np.linspace(0, np.max(xy_points_with_phi[:,1]), 300))
Z = interp_nu(X, Y)
y_edge = interp_edge(X)
mask = Y > y_edge
Z_masked = np.ma.array(Z, mask=mask)

fig7, ax7 = plt.subplots(dpi=600, figsize = (8,6))
#cm6 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='bwr', norm=matplotlib.colors.CenteredNorm(vcenter=0))
cm7 = plt.pcolormesh(X, Y, Z_masked, shading='auto', cmap='viridis')
#ax5.contour(X, Y, Z_masked)
plt.scatter(top_points[:,0], top_points[:,1], c='black', edgecolor='k', cmap='viridis')
#plt.scatter(top_points_2[:,0], top_points_2[:,1], c='grey', edgecolor='k', cmap='viridis')
#cm2 = plt.scatter(xy_points_with_phi[:, 0], xy_points_with_phi[:, 1] ,s=0.1, edgecolor='none')
plt.title("nu distribution")
#plt.colorbar(cm)
plt.colorbar(cm7)

plt.show()
print('final')