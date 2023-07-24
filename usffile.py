import numpy as np
from scipy import special as spec

# r is a numpy array. In nm.
# r is the positions variable.

# # Spherical geometry.
# def Usf(r,params):
    # def sBakshUsf0(r, R):
        # x = R - r
        # first = 0.0
        # second = 0.0
        # for i in range(10):
            # first += 1.0 / R ** i / x ** (10 - i)
            # first += (-1.0) ** i / R ** i / (x - 2.0 * R) ** (10 - i)
            # if i < 4:
                # second += 1.0 / R ** i / x ** (4 - i)
                # second += (-1.0) ** i / R ** i / (x - 2.0 * R) ** (4 - i)
        # Usf = 2.0 / 5.0 * first - second
        # # Has to be multiplied by what precedes it
        # return Usf
    # R = params['poresize[nm]']*0.5
    # sigma_sf = params['sigma_sf[nm]']
    # eps_sf = params['epsilon_sf[K]']
    # eps_ff = params['epsilon_ff[K]']
    # n_s = params['n_s[nm^-2]']
    # # NM to LJ units are with respect to SF parameters
    # R_sf = R / sigma_sf
    # r_sf = r / sigma_sf
    # sUsf = sBakshUsf0(r_sf, R_sf)
    # sUsf = sUsf * (n_s * sigma_sf ** 2) * 2.0 * np.pi * eps_sf #K
    # return sUsf

# # Cylindrical geometry.
# def Usf(r,params):
    # def Usf0(r, R):
        # x = R - r
        # U1 = 63.0 / 32.0 * 1.0 / (x ** 10.0 * (2.0 - x / R) ** 10.0) * spec.hyp2f1(-4.5, -4.5, 1.0, (1.0 - x / R) ** 2)
        # U2 = 3.0 / (x ** 4.0 * (2.0 - x / R) ** 4.0) * spec.hyp2f1(-1.5, -1.5, 1.0, (1.0 - x / R) ** 2)
        # Usf0 = U1 - U2
        # Usf0 *= np.pi ** 2
        # return Usf0
    # n_s = params['n_s[nm^-2]']
    # sigma_sf = params['sigma_sf[nm]']
    # eps_sf = params['epsilon_sf[K]']
    # eps_ff = params['epsilon_sf[K]']
    # R = params['poresize[nm]']*0.5
    # # 1/NM^2 to ssf
    # n_s_lj = n_s * sigma_sf ** 2
    # # NM to LJ units are with respect to SF parameters
    # R_sf = R / sigma_sf
    # r_sf = r / sigma_sf
    # Usf = Usf0(r_sf, R_sf)
    # # Express result in eps_ff units
    # Usf = Usf * eps_sf * n_s_lj
    # return Usf #K

# # Formula from Tjatjopoulos et al. 1988
# def Usf(r,params):
    # def Usf0(r,R):
        # x = R - r
        # U1 = 63.0 / 32.0 * 1.0 / (x ** 10.0 * (2.0 - x / R) ** 10.0) * spec.hyp2f1(-4.5, -4.5, 1.0, (1.0 - x / R) ** 2)
        # U2 = 3.0 / (x ** 4.0 * (2.0 - x / R) ** 4.0) * spec.hyp2f1(-1.5, -1.5, 1.0, (1.0 - x / R) ** 2)
        # Usf0 = U1 - U2
        # Usf0 *= np.pi ** 2
        # return Usf0
    # poreRadius = params['poresize[nm]']*0.5
    # sigma_sf = params['sigma_sf[nm]']
    # sigma_ff = params['sigma_ff[nm]']
    # epsilon_sf = params['epsilon_sf[K]']
    # n_s = params['n_s[nm^-2]']
    # # NM To LJ units with respect to SF parameters
    # n_s *= sigma_sf**2 #Reduced
    # poreRadius *= sigma_ff/sigma_sf #Reduced
    # r *= sigma_ff/sigma_sf #Reduced
    # usfArray = Usf0(r,poreRadius)
    # usfArray *= epsilon_sf*n_s
    # return usfArray #K

# # Square well potential.
# def Usf(r,params):
    # usfArray = np.zeros_like(r)
    # epsilon = params['epsilon_sf[K]']
    # sigmaff = params['sigma_ff[nm]']
    # sigmasf = params['sigma_sf[nm]']
    # deltap = params['sigma2_sf[nm]']*sigmaff
    # poreRadius = params['poresize[nm]']*0.5
    # for i in range(len(r)):
        # if r[i] >= poreRadius-sigmasf*0.5: usfArray[i] = 1e100
        # elif poreRadius-sigmasf*0.5-deltap <= r[i] < poreRadius-sigmasf*0.5: usfArray[i] = -epsilon
    # return usfArray #K

# Square-well with no solid-fluid interactions between the slits.
# def Usf(zArray, params):
#     usfArray = np.zeros_like(zArray)
#     sigmasf = params['sigma_sf[nm]']
#     poreRadius = params['poresize[nm]']*0.5
#     for i in range(len(zArray)):
#         if abs(zArray[i]) >= poreRadius-sigmasf*0.5: usfArray[i] = 1e100
#     return usfArray

# Slit geometry.
def Usf(zArray,params):
    N = 3 # Number of wall layers in each direction
    usfArray = np.zeros_like(zArray)
    eps_fs = params['epsilon_sf[K]']
    sigmasf = params['sigma_sf[nm]']
    W = params['poresize[nm]'] # Pore width (distance between the inner walls)
    n = params['n_s[nm^-2]'] # number of atoms per wall (approximate if not constant)
    d = W/2 # Half pore width
    r_0 = sigmasf
    U = [0] * len(zArray)
    for nw in np.linspace(0,N-1,N):
        T1 = [(((r_0 / (d + (nw*0.335) + z))**10)) for z in zArray]
        T2 = [(((r_0 / (d + (nw*0.335) - z))**10)) for z in zArray]
        T3 = [(((r_0 / (d + (nw*0.335) + z))**4)) for z in zArray]
        T4 = [(((r_0 / (d + (nw*0.335) - z))**4)) for z in zArray]
        U_wall = [(4*np.pi*n*eps_fs*(r_0**2))*((1/5)*(t1+t2)-(1/2)*(t3+t4)) for t1,t2,t3,t4 in zip(T1,T2,T3,T4)]
        U = [u+u_w for u,u_w in zip(U,U_wall)]
    usfArray = np.array(U)
    return usfArray

