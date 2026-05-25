import numpy as np
from numba import njit, jit
from Foreground_models import *
from numba_friendly.stats import normpdf,skewnormpdf,truncnorm_rvs
from numba_friendly.cosmology import angular_diameter_distance,angular_diameter_distance_z1z2
from constants import *

import warnings
from numba.core.errors import NumbaPerformanceWarning
# warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


#* Foreground galaxy population
@njit
def P_psi_etag(eta_g:np.ndarray, 
              logMstar,
              logRe,
              logn,
              m5,
              gamma5,
              zd
              ) -> np.ndarray:
     
    mu_5_0,beta_5,xi_5,sigma_5,mu_gamma_0,beta_gamma,xi_gamma,sigma_gamma = eta_g

    p_zd = P_zd(zd)
    p_m5 = P_m5_logm_logn_logre(m5,logMstar,logRe,logn,mu_5_0,beta_5,xi_5,sigma_5)
    p_gamma5 = P_gamma5_logm_logn_logre(gamma5,logMstar,logRe,logn,mu_gamma_0,beta_gamma,xi_gamma,sigma_gamma)
    p_mstar = P_Mstar(logMstar)
    p_nsers = P_nsers_logm(logn,logMstar)
    p_re = P_Re_logm_logn(logRe,logMstar,logn)

    res = p_m5 * p_gamma5 * p_mstar * p_nsers * p_re * p_zd

    return res

@njit
def Peff_zs_etas(zs,eta_s):
    mu_s,sigma_s = eta_s
    return normpdf(zs, mu_s, sigma_s)

@njit
def zs_generator(eta_s,size):
    mu_s,sigma_s = eta_s
    return np.random.normal(loc=mu_s, scale=sigma_s, size=size)

from astropy.constants import G, c
import h5py
G = G.to('kpc3/ (Msun s2)').value
c = c.to('kpc/s').value
with h5py.File('/Users/liurongfu/tools/dangular_grid.h5','r') as f:
    z_array = f['z'][()]
    D_array = f['Dand'][()]
    comv_array = f['Dcomv'][()]
@njit
def theta_ein(zl,zs,m5,gamma,c = c, G = G):
    Dl = angular_diameter_distance(zl)
    Ds = angular_diameter_distance(zs)
    Dls = angular_diameter_distance_z1z2(zl, zs)
    Sigma_c = c**2/(4*np.pi*G)*Ds/(Dl*Dls)  # Msun/kpc2
    r_ein =  (10**m5 / (np.pi*Sigma_c*5**(3-gamma)))**(1/(gamma -1))  # kpc
    theta_ein = r_ein / Dl * 206265 # arcsec
    theta_ein[np.where(zl >= zs)] = 0
    # theta_ein[np.where(theta_ein > 5)] = 5
    return theta_ein


    


@njit
def Pfind_sigmoid_thetae_etaf(theta_ein,eta_f):
    loga, theta0 = eta_f
    a = 10**loga
    res = 1/(1 + np.exp(-a*(theta_ein - theta0)))
    return res

@njit
def Pfind_guass_thetae_etaf(theta_ein,eta_f):
    mu_thetae, sigma_thetae = eta_f
    res = normpdf(theta_ein, mu_thetae, sigma_thetae)
    return res


@njit
def g_thetae_gamma5(theta_ein,gamma,g_gamma_grids ,cs_over_theta_ein_grid ):
    cs_over_theta_ein = np.interp(gamma, g_gamma_grids, cs_over_theta_ein_grid)
    cs  = np.pi * (cs_over_theta_ein * theta_ein) **2
    return cs


@njit
def draw_sample(size,eta_5, eta_gamma):

    #* seed 
    # np.random.seed(0)
    logMstar_sampled = logMstar_generator(size= size)
    m5_sampled = np.zeros(size)
    gamma5_sampled = np.zeros(size)
    for i in range(size):
        logMstar = logMstar_sampled[i]
        logn = np.random.normal(loc = mu_logn(logMstar),scale= sigma_n)
        logRe = np.random.normal(loc = muR_logm_logn(logMstar, logn), scale= sigma_R)
        m5_sampled[i] = np.random.normal(loc = mu_m5(logMstar, logRe, logn, eta_5), scale = eta_5[-1])
        gamma5_sampled[i] = truncnorm_rvs(1.2,2.8, loc = mu_gamma5(logMstar, logRe, logn, eta_gamma), scale= eta_gamma[-1],size = 1)[0]
    return m5_sampled, gamma5_sampled

@njit
def P_SL_norm_mc(eta,g_gamma_grids,cs_over_theta_ein_grid):
    
    np.random.seed(0)

    eta_g, eta_s, eta_f = eta[0:8], eta[8:10], eta[10:12]
    eta_5,eta_gamma = eta_g[0:4], eta_g[4:8]

    #* inner Monte Carlo integration
    size = 100000
    m5_sampled, gamma5_sampled= draw_sample(size, eta_5, eta_gamma)
    # if np.any(mu_gamma5_sampled < 1.2) or np.any(mu_gamma5_sampled > 2.8):
    #     return 0
    
    zd_sampled = zd_generator(size=size)
    zs_sampled = zs_generator(eta_s, size=size)

    theta_ein_val = theta_ein(zd_sampled, zs_sampled, m5_sampled, gamma5_sampled)
    p_find = Pfind_sigmoid_thetae_etaf(theta_ein_val, eta_f) #* sigmoid selection
    # p_find = Pfind_guass_thetae_etaf(theta_ein_val, eta_f) #* guass selection
    g = g_thetae_gamma5(theta_ein_val, gamma5_sampled, g_gamma_grids, cs_over_theta_ein_grid)
    integrand_samples = p_find * g
    total_integrated = np.mean(integrand_samples)

    if np.abs(total_integrated) < 1e-10:
        # return integrand_samples
        return 0
    else :
        return 1/total_integrated
        # return integrand_samples

@njit
def P_sigmaobs_gamma5_m5(gamma5, m5, sigma_obs, sigma_err, sigma_grid, gamma_grid):
    
    
    if sigma_obs is None:
        res = np.array([1.0], dtype=np.float64)
    else:
        s2 = np.interp(gamma5, gamma_grid, sigma_grid) * 10**m5
        sigma_model = np.sqrt(s2)
        # print(f'sigma_model: {sigma_model}, sigma_obs: {sigma_obs}')
        res = np.atleast_1d(normpdf(sigma_obs, sigma_model, sigma_err)).astype(np.float64)  # 确保类型是 float64
        # res = np.array([1.0], dtype=np.float64)
        
    return res

@njit
def P_mstarobs_obs(logMstar_obs, logMstar_err, logMstar):
    return normpdf(logMstar_obs, logMstar, logMstar_err)

@njit
def integrand( eta,
                logMstar_obs,
                logMstar_err,
                logRe_obs,
                logn_obs,
                sigma_obs,
                zs_obs,
                zd_obs,
                sigma_err,
                theta_ein_obs,
                logMstar,
                gamma5,
                sigma_grid,
                m5_grid,
                dm5_dthetaein_grid,
                gamma_grid,
                cs_over_theta_ein_grid
                ):
    eta_g = eta[0:8]
    eta_s = eta[8:10]
    eta_f = eta[10:12]  

    m5 = np.interp(gamma5, gamma_grid, m5_grid)
    dm5_dthetaein = np.abs(np.interp(gamma5, gamma_grid, dm5_dthetaein_grid))
    
    p_psi = P_psi_etag(eta_g, logMstar, logRe_obs, logn_obs, m5, gamma5, zd_obs)
    p_eff = Peff_zs_etas(zs_obs, eta_s)
    p_find = Pfind_sigmoid_thetae_etaf(theta_ein_obs, eta_f) #* sigmoid selection
    # p_find = Pfind_guass_thetae_etaf(theta_ein_obs, eta_f) #* guass selection
    p_sigma = P_sigmaobs_gamma5_m5(gamma5, m5, sigma_obs, sigma_err, sigma_grid, gamma_grid)
    p_sigma = np.prod(p_sigma)
    p_mstar = P_mstarobs_obs(logMstar_obs, logMstar_err, logMstar)
    # print(f"p_psi: {p_psi}, p_eff: {p_eff}, p_find: {p_find}, p_sigma: {p_sigma}, p_mstar: {p_mstar}, dm5_dthetaein: {dm5_dthetaein}")

    g_gamma_grid = np.linspace(1.2,2.8,81)
    g = g_thetae_gamma5(theta_ein_obs, gamma5, g_gamma_grid, cs_over_theta_ein_grid)

    res = p_psi * p_eff * p_find * p_sigma * p_mstar * dm5_dthetaein * g 

    return res


@njit
def integral( eta,
            logMstar_obs,
            logMstar_err,
            logRe_obs,
            logn_obs,
            sigma_obs,
            zs_obs,
            zd_obs,
            sigma_err,
            theta_ein_obs,
            sigma_grid,
            m5_grid,
            dm5_dthetaein_grid,
            gamma_grid,
            cs_over_theta_ein_grid
            ):
    
    # import time
    logMstar_min,logMstar_max, nlogMstar =  logMstar_obs - 5*logMstar_err,logMstar_obs + 5*logMstar_err,200
    logMstar_ar = np.linspace(logMstar_min, logMstar_max, nlogMstar)

    #* ordinary integration
    gamma5_min, gamma5_max, ngamma5 = 1.2,2.8,200
    gamma5_ar = np.linspace(gamma5_min, gamma5_max, ngamma5)

    #* integration to test small std_gamma
    # s_grid = np.sqrt(sigma_grid * np.power(10, m5_grid))
    # gamma5_high = np.interp(sigma_obs[0] + 5 * sigma_err[0], s_grid, gamma_grid)
    # gamma5_low = np.interp(sigma_obs[0] - 5 * sigma_err[0], s_grid, gamma_grid)

    # gamma5_min = gamma5_high if gamma5_high < gamma5_low else gamma5_low
    # gamma5_max = gamma5_high if gamma5_high > gamma5_low else gamma5_low
    # ngamma5 = 200
    # gamma5_ar = np.linspace(gamma5_min, gamma5_max, ngamma5)

    integrand_grids = np.zeros((nlogMstar, ngamma5))

    for i in range(nlogMstar):
        for j in range(ngamma5):
            logMstar = logMstar_ar[i]
            gamma5 = gamma5_ar[j]
            # print(num_sigma)
            # start = time.time()
            integrand_grids[i,j] = integrand(eta,
                                    logMstar_obs,
                                    logMstar_err,
                                    logRe_obs,
                                    logn_obs,
                                    sigma_obs,
                                    zs_obs,
                                    zd_obs,
                                    sigma_err,
                                    theta_ein_obs,
                                    logMstar,
                                    gamma5,
                                    sigma_grid,
                                    m5_grid,
                                    dm5_dthetaein_grid,
                                    gamma_grid,
                                    cs_over_theta_ein_grid
                                    ) 
            # end = time.time()
            # print(f"Time for integrand computation at i={i}, j={j}: ", end - start)

    gamma5_integrated = np.zeros(nlogMstar)
    for i in range(nlogMstar):
        gamma5_integrated[i] = np.trapezoid(integrand_grids[i,:], gamma5_ar)

    # import time
    # start = time.time()
    total_integrated = np.trapezoid(gamma5_integrated, logMstar_ar)
    # end = time.time()
    # print("Time for integration:", end - start)
    return total_integrated

@njit
def integral_MC( eta,
                logMstar_obs,
            logMstar_err,
            logRe_obs,
            logn_obs,
            sigma_obs,
            zs_obs,
            zd_obs,
            sigma_err,
            theta_ein_obs,
            sigma_grid,
            m5_grid,
            dm5_dthetaein_grid,
            gamma_grid,
            cs_over_theta_ein_grid
            ):
    size = 10000
    logMstar_sampled = logMstar_generator(size= size)



# @jit
def P_d_eta(eta):
    import h5py
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    with h5py.File('./data/cs_grid_power.h5', 'r') as f:
        grp = f['compressed_grids']
        cs_over_theta_ein_grid = np.array(grp['cs_over_theta_ein_grid'][:])

    log_prob = 0

    bounds = [(9,12),(-3,3),(-3,3),(1e-2,0.2),(1.5,2.5),(-3,3),(-3,3),(0,0.5),(1,3),(0,2),(-1,3),(0,3)]

    for i in range(len(eta)):
        if eta[i] < bounds[i][0] or eta[i] > bounds[i][1]:
            return -np.inf
        
    g_gamma_grid = np.linspace(1.2,2.8,81)
    norm = P_SL_norm_mc(eta, g_gamma_grid, cs_over_theta_ein_grid)
    if norm == 0:
        return -np.inf

    with h5py.File('./data/observations_with_m5_grids.hdf5', 'r') as f:
        for name in f.keys():
            # if name == '023817-054555':
            #     continue
            grp = f[name]
            logMstar_obs = grp.attrs['logmchab']
            logMstar_err = grp.attrs['logmchab_err']
            logn_obs = np.log10(grp.attrs['nser'])
            logRe_obs = np.log10(cosmo.kpc_proper_per_arcmin(grp.attrs['zd']).value * grp.attrs['re_arcsec'] / 60.0)
            sigma_obs = grp.attrs['sigma']
            zs_obs = grp.attrs['zs']
            zd_obs = grp.attrs['zd']
            sigma_err = grp.attrs['sigma_err']
            theta_ein_obs = grp.attrs['rein_arcsec']
            # num_sigma = grp.attrs['num_sigma']

            gamma_grid = grp['gamma_grid'][:]
            sigma_grid = grp['s2_grid'][:]
            m5_grid = grp['m5_grid'][:]
            dm5_dthetaein_grid = grp['dm5_dthetaein_grid'][:]

            inte = integral( eta,
                    logMstar_obs,
                    logMstar_err,
                    logRe_obs,
                    logn_obs,
                    sigma_obs,
                    zs_obs,
                    zd_obs,
                    sigma_err,
                    theta_ein_obs,
                    sigma_grid,
                    m5_grid,
                    dm5_dthetaein_grid,
                    gamma_grid,
                    cs_over_theta_ein_grid
                  ) * norm
            # print('Processing ', name, ' integral: ', inte)
            
            if inte <= 0 :
                log_prob += -np.inf
            else:
                log_prob += np.log(inte)
    return log_prob

def P_d_eta_all(eta):
    import h5py
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    with h5py.File('./data/cs_grid_power.h5', 'r') as f:
        grp = f['compressed_grids']
        cs_over_theta_ein_grid = np.array(grp['cs_over_theta_ein_grid'][:])

    log_prob = 0

    bounds = [(9,12),(-3,3),(-3,3),(1e-2,0.2),(1.5,2.5),(-3,3),(-3,3),(0,0.5),(1,3),(0,2),(-1,3),(0,3)]

    for i in range(len(eta)):
        if eta[i] < bounds[i][0] or eta[i] > bounds[i][1]:
            return -np.inf, (None,None)
        
    g_gamma_grid = np.linspace(1.2,2.8,81)
    norm = P_SL_norm_mc(eta, g_gamma_grid, cs_over_theta_ein_grid)
    if norm == 0:
        return -np.inf, (None,None)

    with h5py.File('./data/observations_with_m5_grids_all.hdf5', 'r') as f:
        for name in f.keys():
            grp = f[name]
            logMstar_obs = grp.attrs['logmchab']
            logMstar_err = grp.attrs['logmchab_err']
            logn_obs = np.log10(grp.attrs['nser'])
            logRe_obs = np.log10(cosmo.kpc_proper_per_arcmin(grp.attrs['zd']).value * grp.attrs['re_arcsec'] / 60.0)
            num_sigma = grp.attrs['num_sigma']
            zs_obs = grp.attrs['zs']
            zd_obs = grp.attrs['zd']
            theta_ein_obs = grp.attrs['rein_arcsec']
            # num_sigma = grp.attrs['num_sigma']
            if num_sigma != 0:
                gamma_grid = grp['gamma_grid'][:]
                sigma_grid = grp['s2_grid'][:]
                m5_grid = grp['m5_grid'][:]
                dm5_dthetaein_grid = grp['dm5_dthetaein_grid'][:]

                sigma_obs = grp.attrs['sigma']
                sigma_err = grp.attrs['sigma_err']

                likelihood = integral( eta,
                        logMstar_obs,
                        logMstar_err,
                        logRe_obs,
                        logn_obs,
                        sigma_obs,
                        zs_obs,
                        zd_obs,
                        sigma_err,
                        theta_ein_obs,
                        sigma_grid,
                        m5_grid,
                        dm5_dthetaein_grid,
                        gamma_grid,
                        cs_over_theta_ein_grid
                    )
                inte = likelihood * norm
                blob = (likelihood, norm)
                # print('processing ', name, ' integral: ', inte)
            else:
                gamma_grid = grp['gamma_grid'][:]
                m5_grid = grp['m5_grid'][:]
                dm5_dthetaein_grid = grp['dm5_dthetaein_grid'][:]
                sigma_grid = None
                sigma_obs = None
                sigma_err = None

                likelihood = integral( eta,
                        logMstar_obs,
                        logMstar_err,
                        logRe_obs,
                        logn_obs,
                        sigma_obs,
                        zs_obs,
                        zd_obs,
                        sigma_err,
                        theta_ein_obs,
                        sigma_grid,
                        m5_grid,
                        dm5_dthetaein_grid,
                        gamma_grid,
                        cs_over_theta_ein_grid
                    )
                inte = likelihood * norm
                blob = (likelihood, norm)
            
                # print('processing ', name, ' integral: ', inte)
            if inte <= 0 :
                log_prob += -np.inf
            else:
                log_prob += np.log(inte)
    return log_prob, blob

def P_d_eta_mock(eta):
    import h5py
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    with h5py.File('./data/cs_grid_power.h5', 'r') as f:
        grp = f['compressed_grids']
        cs_over_theta_ein_grid = np.array(grp['cs_over_theta_ein_grid'][:])

    log_prob = 0

    bounds = [(9,12),(-3,3),(-3,3),(1e-3,0.2),(1.5,2.5),(-3,3),(-3,3),(0,0.5),(1,3),(0,2),(-1,3),(0,3)]

    for i in range(len(eta)):
        if eta[i] < bounds[i][0] or eta[i] > bounds[i][1]:
            return -np.inf,(None,None)
        
    g_gamma_grid = np.linspace(1.2,2.8,81)
    norm = P_SL_norm_mc(eta, g_gamma_grid, cs_over_theta_ein_grid)
    if norm == 0:
        return -np.inf,(None,None)

    with h5py.File('./data/mock_datas/observations_mock_with_m5_grids.hdf5', 'r') as f:
        for name in f.keys():
            grp = f[name]
            logMstar_obs = grp.attrs['logmchab']
            logMstar_err = grp.attrs['logmchab_err']
            logn_obs = np.log10(grp.attrs['nser'])
            logRe_obs = np.log10(cosmo.kpc_proper_per_arcmin(grp.attrs['zd']).value * grp.attrs['re_arcsec'] / 60.0)
            num_sigma = grp.attrs['num_sigma']
            zs_obs = grp.attrs['zs']
            zd_obs = grp.attrs['zd']
            theta_ein_obs = grp.attrs['rein_arcsec']
            # num_sigma = grp.attrs['num_sigma']
            if num_sigma != 0:
                gamma_grid = grp['gamma_grid'][:]
                sigma_grid = grp['s2_grid'][:]
                m5_grid = grp['m5_grid'][:]
                dm5_dthetaein_grid = grp['dm5_dthetaein_grid'][:]

                sigma_obs = grp.attrs['sigma']
                sigma_err = grp.attrs['sigma_err']

                likelihood = integral( eta,
                        logMstar_obs,
                        logMstar_err,
                        logRe_obs,
                        logn_obs,
                        sigma_obs,
                        zs_obs,
                        zd_obs,
                        sigma_err,
                        theta_ein_obs,
                        sigma_grid,
                        m5_grid,
                        dm5_dthetaein_grid,
                        gamma_grid,
                        cs_over_theta_ein_grid
                    )
                inte = likelihood * norm
                blob = (likelihood, norm)
                # print('processing ', name, ' integral: ', inte)
            else:
                gamma_grid = grp['gamma_grid'][:]
                m5_grid = grp['m5_grid'][:]
                dm5_dthetaein_grid = grp['dm5_dthetaein_grid'][:]
                sigma_grid = None
                sigma_obs = None
                sigma_err = None

                likelihood = integral( eta,
                        logMstar_obs,
                        logMstar_err,
                        logRe_obs,
                        logn_obs,
                        sigma_obs,
                        zs_obs,
                        zd_obs,
                        sigma_err,
                        theta_ein_obs,
                        sigma_grid,
                        m5_grid,
                        dm5_dthetaein_grid,
                        gamma_grid,
                        cs_over_theta_ein_grid
                    )
                inte = likelihood * norm
                blob = (likelihood,norm)
            
                # print('processing ', name, ' integral: ', inte)
            if inte <= 0 :
                log_prob += -np.inf
            else:
                log_prob += np.log(inte)
    return log_prob, blob


@njit
def draw_sample_test(size,eta_5, eta_gamma):
    np.random.seed(0)
    logMstar_sampled = logMstar_generator(size= size)
    # mu_logn_sampled = mu_logn(logMstar_sampled)
    # mu_logre_sampled = muR_logm_logn(logMstar_sampled, mu_logn_sampled)
    # mu_m5_sampled = mu_m5(logMstar_sampled, mu_logre_sampled, mu_logn_sampled,eta_5)
    # mu_gamma5_sampled = mu_gamma5(logMstar_sampled, mu_logre_sampled, mu_logn_sampled, eta_gamma)
    logn_sampled = np.zeros(size)
    logRe_sampled = np.zeros(size)
    m5_sampled = np.zeros(size)
    gamma5_sampled = np.zeros(size)
    for i in range(size):
        logMstar = logMstar_sampled[i]
        logn = np.random.normal(loc = mu_logn(logMstar),scale= sigma_n)
        logRe = np.random.normal(loc = muR_logm_logn(logMstar, logn), scale= sigma_R)
        logn_sampled[i] = logn
        logRe_sampled[i] = logRe
        m5_sampled[i] = np.random.normal(loc = mu_m5(logMstar, logRe, logn, eta_5), scale = eta_5[-1])
        gamma5_sampled[i] = truncnorm_rvs(1.2,2.8, loc = mu_gamma5(logMstar, logRe, logn, eta_gamma), scale= eta_gamma[-1],size = 1)[0]
        # logMstar_sample = logMstar_sampled[i]
        # logn_sampled[i] = np.random.normal(loc = mu_logn_sampled[i], scale= sigma_n)
        # logRe_sampled[i] = np.random.normal(loc = mu_logre_sampled[i], scale= sigma_R)
        # m5_sampled[i] = np.random.normal(loc = mu_m5_sampled[i], scale = eta_5[-1])
        # gamma5_sampled[i] = np.random.normal(loc = mu_gamma5_sampled[i], scale= eta_gamma[-1])
        # logn_sampled[i] = logn_generator(logMstar_sampled[i], size=1)[0]
        # logRe_sampled[i] = logRe_generator(logMstar_sampled[i], logn_sampled[i], size=1)[0]
        # m5_sampled[i] = m5_generator(eta_5, logMstar_sampled[i], logRe_sampled[i], logn_sampled[i], size=1)[0]
        # gamma5_sampled[i] = gamma5_generator(eta_gamma, logMstar_sampled[i], logRe_sampled[i], logn_sampled[i], size=1)[0]
    return logMstar_sampled, logn_sampled, logRe_sampled, m5_sampled, gamma5_sampled

@njit
def draw_sample_test2(size,eta_5, eta_gamma):

    #* seed 
    # np.random.seed(0)
    logMstar_sampled = logMstar_generator(size= size)
    m5_sampled = np.zeros(size)
    gamma5_sampled = np.zeros(size)
    mu_gamma5_sampled = np.zeros(size)
    for i in range(size):
        logMstar = logMstar_sampled[i]
        logn = np.random.normal(loc = mu_logn(logMstar),scale= sigma_n)
        logRe = np.random.normal(loc = muR_logm_logn(logMstar, logn), scale= sigma_R)
        m5_sampled[i] = np.random.normal(loc = mu_m5(logMstar, logRe, logn, eta_5), scale = eta_5[-1])
        # gamma5_sampled[i] = np.random.normal(loc = mu_gamma5(logMstar, logRe, logn, eta_gamma), scale= eta_gamma[-1])
        mu_gamma5_sampled[i] = mu_gamma5(logMstar, logRe, logn, eta_gamma)
        gamma5_sampled[i] = np.random.normal(loc = mu_gamma5(logMstar, logRe, logn, eta_gamma), scale= eta_gamma[-1])
    return m5_sampled, gamma5_sampled,mu_gamma5_sampled

@njit
def P_SL_norm_mc_test(eta,g_gamma_grids,cs_over_theta_ein_grid):
    
    np.random.seed(0)

    eta_g, eta_s, eta_f = eta[0:8], eta[8:10], eta[10:12]
    eta_5,eta_gamma = eta_g[0:4], eta_g[4:8]

    #* inner Monte Carlo integration
    size = 100000
    m5_sampled, gamma5_sampled,mu_gamma5_sampled = draw_sample_test2(size, eta_5, eta_gamma)
    # if np.any(mu_gamma5_sampled < 1.2) or np.any(mu_gamma5_sampled > 2.8):
    #     return 0
    zd_sampled = zd_generator(size=size)
    zs_sampled = zs_generator(eta_s, size=size)

    theta_ein_val = theta_ein(zd_sampled, zs_sampled, m5_sampled, gamma5_sampled)
    p_find = Pfind_guass_thetae_etaf(theta_ein_val, eta_f)
    g = g_thetae_gamma5(theta_ein_val, gamma5_sampled, g_gamma_grids, cs_over_theta_ein_grid)
    integrand_samples = p_find * g
    total_integrated = np.mean(integrand_samples)

    if np.abs(total_integrated) < 1e-10:
        return 0
    else :
        return 1/total_integrated


# @njit
# def P_SL_norm(eta,logMstar,logRe,logn,g_gamma_grids,cs_over_theta_ein_grid):
#     eta_g, eta_s, eta_f = eta[0:8], eta[8:10], eta[10:12]
#     size = 20
#     m5_grid = np.linspace(10,12,size)
#     gamma_grid = np.linspace(1.2,2.8,size)
#     zd_grid = np.linspace(0.2,0.8,size)
#     mu_s,sigma_s = eta_s
#     zs_grid = np.linspace(0.81, mu_s + 4*sigma_s,size)
#     logMstar_grid = np.linspace(10,13,size)
#     logRe_grid = np.linspace(-0.5,2,size)
#     logn_grid = np.linspace(1,15,size)

#     integrand_grids = np.zeros((len(m5_grid), len(gamma_grid), len(zd_grid), len(zs_grid), len(logMstar_grid), len(logRe_grid), len(logn_grid)))

#     for i in range(len(m5_grid)):
#         for j in range(len(gamma_grid)):
#             for k in range(len(zd_grid)):
#                 for l in range(len(zs_grid)):
#                     for m in range(len(logMstar_grid)):
#                         for n in range(len(logRe_grid)):
#                             for o in range(len(logn_grid)):
#                                 m5 = m5_grid[i]
#                                 gamma5 = gamma_grid[j]
#                                 zd = zd_grid[k]
#                                 zs = zs_grid[l]
#                                 logMstar = logMstar_grid[m]
#                                 logRe = logRe_grid[n]
#                                 logn = logn_grid[o]

#                                 theta_ein_val = theta_ein(zd, zs, m5, gamma5)

#                                 p_psi = P_psi_etag(eta_g, logMstar, logRe, logn, m5, gamma5, zd)
#                                 p_eff = Peff_zs_etas(zs, eta_s)
#                                 p_find = Pfind_thetae_etaf(theta_ein_val, eta_f)
#                                 g = g_thetae_gamma5(theta_ein_val, gamma5, g_gamma_grids, cs_over_theta_ein_grid)

#                                 integrand_grids[i,j,k,l,m,n,o] = p_psi * p_eff * p_find * g
    
#     logn_integrated = np.zeros((len(m5_grid), len(gamma_grid), len(zd_grid), len(zs_grid), len(logMstar_grid), len(logRe_grid)))
#     for i in range(len(m5_grid)):
#         for j in range(len(gamma_grid)):
#             for k in range(len(zd_grid)):
#                 for l in range(len(zs_grid)):
#                     for m in range(len(logMstar_grid)):
#                         for n in range(len(logRe_grid)):
#                             logn_integrated[i,j,k,l,m,n] = np.trapezoid(integrand_grids[i,j,k,l,m,n,:], logn_grid)

#     logRe_integrated = np.zeros((len(m5_grid), len(gamma_grid), len(zd_grid), len(zs_grid), len(logMstar_grid)))
#     for i in range(len(m5_grid)):
#         for j in range(len(gamma_grid)):
#             for k in range(len(zd_grid)):
#                 for l in range(len(zs_grid)):
#                     for m in range(len(logMstar_grid)):
#                         logRe_integrated[i,j,k,l,m] = np.trapezoid(logn_integrated[i,j,k,l,m,:], logRe_grid)

#     logMstar_integrated = np.zeros((len(m5_grid), len(gamma_grid), len(zd_grid), len(zs_grid)))
#     for i in range(len(m5_grid)):
#         for j in range(len(gamma_grid)):
#             for k in range(len(zd_grid)):
#                 for l in range(len(zs_grid)):
#                     logMstar_integrated[i,j,k,l] = np.trapezoid(logRe_integrated[i,j,k,l,:], logMstar_grid)
    
#     zs_integrated = np.zeros((len(m5_grid), len(gamma_grid), len(zd_grid)))
#     for i in range(len(m5_grid)):
#         for j in range(len(gamma_grid)):
#             for k in range(len(zd_grid)):
#                 zs_integrated[i,j,k] = np.trapezoid(logMstar_integrated[i,j,k,:], zs_grid)
    
#     zd_integrated = np.zeros((len(m5_grid), len(gamma_grid)))
#     for i in range(len(m5_grid)):
#         for j in range(len(gamma_grid)):
#             zd_integrated[i,j] = np.trapezoid(zs_integrated[i,j,:], zd_grid)

#     gamma_integrated = np.zeros(len(m5_grid))
#     for i in range(len(m5_grid)):
#         gamma_integrated[i] = np.trapezoid(zd_integrated[i,:], gamma_grid)

#     total_integrated = np.trapezoid(gamma_integrated, m5_grid)
#     return total_integrated