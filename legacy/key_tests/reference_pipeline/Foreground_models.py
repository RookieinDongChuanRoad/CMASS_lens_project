import numpy as np
# from scipy.stats import skewnorm,norm
from numba_friendly.stats import *
from numba import njit,jit

import warnings
from numba.core.errors import NumbaPerformanceWarning
# warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

@njit
def P_zd(zd:float) -> float:

    mu_d = 0.558
    sigma_d = 0.085

    res = normpdf(zd, mu_d, sigma_d)
    return res

@njit 
def zd_generator(size:int) :
    mu_d = 0.558
    sigma_d = 0.085

    res = np.random.normal(loc = mu_d, scale = sigma_d, size = size)
    return res

@njit
def P_Mstar(logMstar:float) :

    mu_star = 11.249
    sigma_star = 0.285
    alpha_star = 10**0.43

    res = skewnormpdf(logMstar,loc = mu_star, scale = sigma_star, alpha = alpha_star)

    return res

@njit
def logMstar_generator(size:int):
    
    mu_star = 11.249
    sigma_star = 0.285
    alpha_star = 10**0.43

    res = skewnormal_rvs(loc = mu_star, scale = sigma_star, alpha = alpha_star, size = size)
    return res
    


@njit
def mu_logn(logMstar:float|np.ndarray) -> float|np.ndarray:
        
    mu_n0 = 0.704 
    beta_n = 0.464

    return mu_n0 + beta_n*(logMstar - 11.4)

@njit
def P_nsers_logm(logn:float|np.ndarray,logMstar:float|np.ndarray) -> float|np.ndarray:
    
    
    
    sigma_n = 0.163

    res = normpdf(logn, mu_logn(logMstar), sigma_n)

    return res

@njit 
def logn_generator(logMstar,size):

    sigma_n = 0.163

    mu_logn_val = mu_logn(logMstar)
    res = np.random.normal(loc = mu_logn_val, scale = sigma_n, size = size)
    return res

@njit
def muR_logm_logn(logMstar:float|np.ndarray, logn:float|np.ndarray) -> float|np.ndarray:
        
    mu_r0 = 0.817
    beta_r = 1.184
    xi_r = 0.383

    return mu_r0 + beta_r*(logMstar - 11.4) + xi_r*(logn- np.log10(4))

@njit
def P_Re_logm_logn(logre:float|np.ndarray,logMstar:float|np.ndarray, logn:float|np.ndarray) -> float|np.ndarray:    
    #* Re in kpc
    sigma_R = 0.133

    res = normpdf(logre, muR_logm_logn(logMstar,logn), sigma_R)

    return res

@njit
def logRe_generator(logMstar, logn, size:int):
    sigma_R = 0.133

    mu_logRe = muR_logm_logn(logMstar,logn)

    res = np.random.normal(loc = mu_logRe, scale = sigma_R, size = size)

    return res
@njit
def mu_m5(logMstar,logRe,logn,eta_5):
    
    mu_5_0 ,beta_5,xi_5,sigma_5 = eta_5

    mu_m5 = mu_5_0 + beta_5*(logMstar - 11.4) + xi_5*(logRe - muR_logm_logn(logMstar,logn))

    return mu_m5

@njit
def P_m5_logm_logn_logre(m5,logMstar,logRe,logn,mu_5_0,beta_5,xi_5,sigma_5):

    mu_m5 = mu_5_0 + beta_5*(logMstar - 11.4) + xi_5*(logRe - muR_logm_logn(logMstar,logn))
    
    res = normpdf(m5, mu_m5, sigma_5)
    return res

@njit
def m5_generator(eta_5, logMstar, logRe, logn, size:int):

    mu_5_0 ,beta_5,xi_5,sigma_5 = eta_5

    mu_m5 = mu_5_0 + beta_5*(logMstar - 11.4) + xi_5*(logRe - muR_logm_logn(logMstar,logn))

    res = np.random.normal(loc = mu_m5, scale = sigma_5, size = size)
    return res

@njit
def mu_gamma5(logMstar,logRe,logn,eta_gamma):
    mu_gamma_0 ,beta_gamma,xi_gamma,sigma_gamma = eta_gamma

    mu_gamma5 = mu_gamma_0 + beta_gamma*(logMstar - 11.4) + xi_gamma*(logRe - muR_logm_logn(logMstar,logn))

    return mu_gamma5

@njit
def P_gamma5_logm_logn_logre(gamma5,logMstar,logRe,logn,mu_gamma_0,beta_gamma,xi_gamma,sigma_gamma):

    mu_gamma5 = mu_gamma_0 + beta_gamma*(logMstar - 11.4) + xi_gamma*(logRe - muR_logm_logn(logMstar,logn))
    
    res = normpdf(gamma5, mu_gamma5, sigma_gamma)

    return res

@njit 
def gamma5_generator(eta_gamma, logMstar, logRe, logn, size:int):

    mu_gamma_0 ,beta_gamma,xi_gamma,sigma_gamma = eta_gamma

    mu_gamma5 = mu_gamma_0 + beta_gamma*(logMstar - 11.4) + xi_gamma*(logRe - muR_logm_logn(logMstar,logn))

    res = np.random.normal(loc = mu_gamma5, scale = sigma_gamma, size = size)

    return res





    
        
        


    
    




