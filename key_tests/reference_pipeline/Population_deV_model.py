import numpy as np
from numba import njit, jit
from Foreground_deV_models import *
from numba_friendly.stats import normpdf,skewnormpdf,truncnorm_rvs
from numba_friendly.cosmology import angular_diameter_distance,angular_diameter_distance_z1z2
from constants_deV import *

@njit
def P_psi_etag(eta_g:np.ndarray, 
              logMstar,
              logRe,
              m5,
              gamma5,
              zd
              ):
    
    mu_5_0,beta_5,xi_5,sigma_5,mu_gamma_0,beta_gamma,xi_gamma,sigma_gamma = eta_g

    p_zd = P_zd(zd)
    p_m5 = P_m5_logm_logre(m5,logMstar,logRe,mu_5_0,beta_5,xi_5,sigma_5)
    p_gamma5 = P_gamma5_logm_logre(gamma5,logMstar,logRe,mu_gamma_0,beta_gamma,xi_gamma,sigma_gamma)
    p_mstar = P_Mstar(logMstar)
    p_re = P_Re_logm(logRe,logMstar)

    res = p_zd * p_m5 * p_gamma5 * p_mstar * p_re

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
    mu_theta, sigma_theta = eta_f
    res = normpdf(theta_ein, mu_theta, sigma_theta)
    return res


@njit
def g_thetae_gamma5(theta_ein,gamma,g_gamma_grids ,cs_over_theta_ein_grid ):
    cs_over_theta_ein = np.interp(gamma, g_gamma_grids, cs_over_theta_ein_grid)
    cs  = np.pi * (cs_over_theta_ein * theta_ein) **2
    return cs

@njit
def draw_sample(size,eta_5,eta_gamma):
    logMstar_sampled = logMstar_generator(size)
    m5_sampled = np.zeros(size)
    gamma5_sampled = np.zeros(size)
    for i in range(size):
        logMstar = logMstar_sampled[i]
        logRe = np.random.normal(loc = muR_logm(logMstar),scale=sigma_R)
        m5_sampled[i] = np.random.normal(loc=mu_m5(logMstar,logRe,eta_5), scale=eta_5[-1])
        gamma5_sampled[i] = truncnorm_rvs(1.2,2.8,loc = mu_gamma5(logMstar,logRe,eta_gamma),scale = eta_gamma[-1], size= 1)[0]
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
        res = np.atleast_1d(normpdf(sigma_obs, sigma_model, sigma_err)).astype(np.float64)  # 确保类型是 float64
        # res = np.array([1.0], dtype=np.float64)
    return res

@njit
def P_mstarobs_obs(logMstar_obs, logMstar_err, logMstar):
    return normpdf(logMstar_obs, logMstar, logMstar_err)

@njit
def integrand(  eta,
                logMstar_obs,
                logMstar_err,
                logRe_obs,
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
    
    eta_g, eta_s, eta_f = eta[0:8], eta[8:10], eta[10:12]

    m5 = np.interp(gamma5, gamma_grid, m5_grid)
    dm5_dthetaein = np.abs(np.interp(gamma5,gamma_grid,dm5_dthetaein_grid))

    p_psi = P_psi_etag(eta_g, logMstar, logRe_obs, m5, gamma5, zd_obs)
    p_eff = Peff_zs_etas(zs_obs, eta_s)
    p_find = Pfind_sigmoid_thetae_etaf(theta_ein_obs, eta_f) #* sigmoid selection
    # p_find = Pfind_guass_thetae_etaf(theta_ein_obs, eta_f) #* guass selection
    p_sigmaobs = P_sigmaobs_gamma5_m5(gamma5, m5, sigma_obs, sigma_err, sigma_grid, gamma_grid)
    p_sigmaobs = np.prod(p_sigmaobs)
    p_mstar = P_mstarobs_obs(logMstar_obs, logMstar_err, logMstar)

    g_gamma_grid = np.linspace(1.2,2.8,81)
    g = g_thetae_gamma5(theta_ein_obs, gamma5, g_gamma_grid ,cs_over_theta_ein_grid )

    res = p_psi * p_eff * p_find * p_sigmaobs * p_mstar * g * dm5_dthetaein

    return res

@njit
def integral(  eta,
                logMstar_obs,
                logMstar_err,
                logRe_obs,
                sigma_obs,
                zs_obs,
                zd_obs,
                sigma_err,
                theta_ein_obs,
                sigma_grid,
                m5_grid,
                dm5_dthetaein_grid,
                gamma_grid,
                cs_over_theta_ein_grid,
                ):
    
    logMstar_min,logMstar_max, nlogMstar =  logMstar_obs - 5*logMstar_err,logMstar_obs + 5*logMstar_err,200
    logMstar_ar = np.linspace(logMstar_min, logMstar_max, nlogMstar)

    gamma5_min, gamma5_max, ngamma5 = 1.2,2.8,200
    gamma5_ar = np.linspace(gamma5_min, gamma5_max, ngamma5)

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
            integrand_grids[i,j] = integrand(  eta,
                                              logMstar_obs,
                                              logMstar_err,
                                              logRe_obs,
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
    gamma5_integrated = np.zeros(nlogMstar)
    for i in range(nlogMstar):
        gamma5_integrated[i] = np.trapezoid(integrand_grids[i,:], gamma5_ar)

    total_integrated = np.trapezoid(gamma5_integrated, logMstar_ar) 

    return total_integrated

def P_d_eta(eta):
    import h5py
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    with h5py.File('./data/cs_grid_power.h5', 'r') as f:
        grp = f['compressed_grids']
        cs_over_theta_ein_grid = np.array(grp['cs_over_theta_ein_grid'][:])

    log_prob = 0

    bounds = [(9,12),(-3,3),(-3,3),(0,0.2),(1.5,2.5),(-3,3),(-3,3),(0,0.5),(1,3),(0,2),(-1,3),(0,3)]

    for i in range(len(eta)):
        if eta[i] < bounds[i][0] or eta[i] > bounds[i][1]:
            return -np.inf
        
    g_gamma_grid = np.linspace(1.2,2.8,81)
    norm = P_SL_norm_mc(eta, g_gamma_grid, cs_over_theta_ein_grid)
    if norm == 0:
        return -np.inf
    
    with h5py.File('./data/observations_deV_with_m5_grids.hdf5', 'r') as f:
        for name in f.keys():
            # if name == '023817-054555':
            #     continue
            grp = f[name]
            logMstar_obs = grp.attrs['logmchab_deV']
            logMstar_err = grp.attrs['logmchab_err']
            logRe_obs = np.log10(cosmo.kpc_proper_per_arcmin(grp.attrs['zd']).value * grp.attrs['reff_deV'] / 60.0)
            num_sigma = grp.attrs['num_sigma']
            zs_obs = grp.attrs['zs']
            zd_obs = grp.attrs['zd']
            theta_ein_obs = grp.attrs['rein_arcsec']
            gamma_grid = grp['gamma_grid'][:]
            m5_grid = grp['m5_grid'][:]
            dm5_dthetaein_grid = grp['dm5_dthetaein_grid'][:]

            if num_sigma != 0:
                sigma_grid = grp['s2_grid'][:]
                sigma_obs = grp.attrs['sigma']
                sigma_err = grp.attrs['sigma_err']
            else:
                sigma_grid = None
                sigma_obs = None
                sigma_err = None

            inte = integral( eta,
                    logMstar_obs,
                    logMstar_err,
                    logRe_obs,
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
            
            if inte <= 0 :
                log_prob += -np.inf
            else:
                log_prob += np.log(inte)
    return log_prob

def P_d_eta_sigma(eta):
    import h5py
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    with h5py.File('./data/cs_grid_power.h5', 'r') as f:
        grp = f['compressed_grids']
        cs_over_theta_ein_grid = np.array(grp['cs_over_theta_ein_grid'][:])

    log_prob = 0

    bounds = [(9,12),(-3,3),(-3,3),(0,0.2),(1.5,2.5),(-3,3),(-3,3),(0,0.5),(1,3),(0,2),(-1,3),(0,3)]

    for i in range(len(eta)):
        if eta[i] < bounds[i][0] or eta[i] > bounds[i][1]:
            return -np.inf
        
    g_gamma_grid = np.linspace(1.2,2.8,81)
    norm = P_SL_norm_mc(eta, g_gamma_grid, cs_over_theta_ein_grid)
    if norm == 0:
        return -np.inf
    
    with h5py.File('./data/observations_deV_with_m5_grids.hdf5', 'r') as f:
        for name in f.keys():
            # if name == '023817-054555':
            #     continue
            grp = f[name]
            logMstar_obs = grp.attrs['logmchab_deV']
            logMstar_err = grp.attrs['logmchab_err']
            logn_obs = np.log10(grp.attrs['nser'])
            logRe_obs = np.log10(cosmo.kpc_proper_per_arcmin(grp.attrs['zd']).value * grp.attrs['reff_deV'] / 60.0)
            num_sigma = grp.attrs['num_sigma']
            zs_obs = grp.attrs['zs']
            zd_obs = grp.attrs['zd']
            theta_ein_obs = grp.attrs['rein_arcsec']
            # num_sigma = grp.attrs['num_sigma']
            gamma_grid = grp['gamma_grid'][:]
            m5_grid = grp['m5_grid'][:]
            dm5_dthetaein_grid = grp['dm5_dthetaein_grid'][:]

            if num_sigma != 0:
                sigma_grid = grp['s2_grid'][:]
                sigma_obs = grp.attrs['sigma']
                sigma_err = grp.attrs['sigma_err']
            else:
                continue
                sigma_grid = None
                sigma_obs = None
                sigma_err = None

            inte = integral( eta,
                    logMstar_obs,
                    logMstar_err,
                    logRe_obs,
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
            
            if inte <= 0 :
                log_prob += -np.inf
            else:
                log_prob += np.log(inte)
    return log_prob       

def P_d_eta_mock(eta):
    import h5py
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    with h5py.File('./data/cs_grid_power.h5', 'r') as f:
        grp = f['compressed_grids']
        cs_over_theta_ein_grid = np.array(grp['cs_over_theta_ein_grid'][:])

    log_prob = 0

    bounds = [(9,12),(-3,3),(-3,3),(0,0.2),(1.5,2.5),(-3,3),(-3,3),(0,0.5),(1,3),(0,2),(-1,3),(0,3)]

    for i in range(len(eta)):
        if eta[i] < bounds[i][0] or eta[i] > bounds[i][1]:
            return -np.inf
        
    g_gamma_grid = np.linspace(1.2,2.8,81)
    norm = P_SL_norm_mc(eta, g_gamma_grid, cs_over_theta_ein_grid)
    if norm == 0:
        return -np.inf
    
    with h5py.File('./data/mock_datas/observations_deV_mock_with_m5_grids.hdf5', 'r') as f:
        for name in f.keys():
            # if name == '023817-054555':
            #     continue
            grp = f[name]
            logMstar_obs = grp.attrs['logmchab']
            logMstar_err = grp.attrs['logmchab_err']
            logRe_obs = np.log10(cosmo.kpc_proper_per_arcmin(grp.attrs['zd']).value * grp.attrs['re_arcsec'] / 60.0)
            num_sigma = grp.attrs['num_sigma']
            zs_obs = grp.attrs['zs']
            zd_obs = grp.attrs['zd']
            theta_ein_obs = grp.attrs['rein_arcsec']
            # num_sigma = grp.attrs['num_sigma']
            gamma_grid = grp['gamma_grid'][:]
            m5_grid = grp['m5_grid'][:]
            dm5_dthetaein_grid = grp['dm5_dthetaein_grid'][:]

            if num_sigma != 0:
                sigma_grid = grp['s2_grid'][:]
                sigma_obs = grp.attrs['sigma']
                sigma_err = grp.attrs['sigma_err']
            else:
                sigma_grid = None
                sigma_obs = None
                sigma_err = None

            inte = integral( eta,
                    logMstar_obs,
                    logMstar_err,
                    logRe_obs,
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
            if inte <= 0 :
                # if norm > 0:
                
                log_prob += -np.inf
            else:
                log_prob += np.log(inte)
    return log_prob

@njit
def draw_sample_test(size,eta_5,eta_gamma):
    np.random.seed(0)
    logMstar_sampled = logMstar_generator(size)
    logRe_sampled = np.zeros(size)
    # mu_gamma5_sampled = np.zeros(size)
    m5_sampled = np.zeros(size)
    gamma5_sampled = np.zeros(size)
    for i in range(size):
        logMstar = logMstar_sampled[i]
        logRe = np.random.normal(loc = muR_logm(logMstar),scale=sigma_R)
        logRe_sampled[i] = logRe
        m5_sampled[i] = np.random.normal(loc=mu_m5(logMstar,logRe,eta_5), scale=eta_5[-1])
        gamma5_sampled[i] = truncnorm_rvs(1.2,2.8,loc = mu_gamma5(logMstar,logRe,eta_gamma),scale = eta_gamma[-1], size= 1)[0]
        # mu_gamma5_sampled[i] = mu_gamma5(logMstar,logRe,eta_gamma)
    return m5_sampled, gamma5_sampled, logRe_sampled,logMstar_sampled