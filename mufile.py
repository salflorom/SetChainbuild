import numpy as np

def Mu(P,params):
    temp = params['externaltemperature[K]']
    eps_ff = params['epsilon_ff[K]']
    mu0 = -11.573943*eps_ff #K
    p_small = np.linspace(0.0, 0.1, 10)[1:]
    p_rel_min = 0.1
    p_rel = np.concatenate(([1e-5], p_small, np.linspace(0.0, 1.0, 21)[1:]))
    p_rel = np.sort(np.unique(p_rel))
    mu = mu0+temp*np.log(p_rel) #K
    return mu
