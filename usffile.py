import numpy as np

# r is a numpy array.
def Usf(radius,params):
    usfArray = np.zeros_like(radius)
    epsilon = params['epsilon_sf[K]']
    sigma1 = params['sigma_sf[nm]']
    sigma2 = params['sigma2_sf[nm]']
    for i in range(len(radius)):
        if sigma2 <= radius[i]: usfArray[i] = 1e100
        elif sigma1 <= radius[i] < sigma2: usfArray[i] = -1
    return usfArray*epsilon #K
