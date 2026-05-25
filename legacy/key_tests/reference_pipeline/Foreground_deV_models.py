import numpy as np
# from scipy.stats import skewnorm,norm
from numba_friendly.stats import *
from numba import njit,jit

@njit 
def P_zd(zd):
    mu_d = 0.558
    sigma_d = 0.085

    res = normpdf(zd, mu_d, sigma_d)
    return res


@njit 
def zd_generator(size):
    mu_d = 0.558
    sigma_d = 0.085

    res = np.random.normal(loc = mu_d, scale = sigma_d, size = size)
    return res

@njit
def P_Mstar(logMstar):

    mu_star = 11.252
    sigma_star = 0.202
    alpha_star = 10**0.17

    res = skewnormpdf(logMstar,loc = mu_star, scale = sigma_star, alpha = alpha_star)

    return res

@njit
def logMstar_generator(size):
    
    mu_star = 11.252
    sigma_star = 0.202
    alpha_star = 10**0.17

    res = skewnormal_rvs(loc = mu_star, scale = sigma_star, alpha = alpha_star, size = size)
    return res

@njit 
def muR_logm(logMstar):

    mu_r0 = 0.774
    beta_r = 0.977

    return mu_r0 + beta_r*(logMstar - 11.4)

@njit
def P_Re_logm(logRe,logMstar):

    sigma_R = 0.112

    res = normpdf(logRe, muR_logm(logMstar), sigma_R)

    return res

@njit 
def logRe_generator(logMstar,size):

    mu_r0 = 0.774
    beta_r = 0.977
    sigma_R = 0.112

    mu_re = mu_r0 + beta_r*(logMstar - 11.4)

    res = np.random.normal(loc = mu_re, scale = sigma_R, size = size)
    return res

@njit
def mu_m5(logMstar,logRe,eta_5):
    
    mu_5_0 ,beta_5,xi_5,sigma_5 = eta_5

    mu_m5 = mu_5_0 + beta_5*(logMstar - 11.4) + xi_5*(logRe - muR_logm(logMstar))

    return mu_m5

@njit
def P_m5_logm_logre(m5,logMstar,logRe,mu_5_0,beta_5,xi_5,sigma_5):

    mu_m5 = mu_5_0 + beta_5*(logMstar - 11.4) + xi_5*(logRe - muR_logm(logMstar))

    res = normpdf(m5, mu_m5, sigma_5)
    return res

@njit
def m5_generator(eta_5, logMstar, logRe, size:int):

    mu_5_0 ,beta_5,xi_5,sigma_5 = eta_5

    mu_m5 = mu_5_0 + beta_5*(logMstar - 11.4) + xi_5*(logRe - muR_logm(logMstar))

    res = np.random.normal(loc = mu_m5, scale = sigma_5, size = size)
    return res

@njit
def mu_gamma5(logMstar,logRe, eta_gamma):
    mu_gamma_0 ,beta_gamma,xi_gamma,sigma_gamma = eta_gamma

    mu_gamma5 = mu_gamma_0 + beta_gamma*(logMstar - 11.4) + xi_gamma*(logRe - muR_logm(logMstar))

    return mu_gamma5

@njit
def P_gamma5_logm_logre(gamma5,logMstar,logRe, mu_gamma_0,beta_gamma,xi_gamma,sigma_gamma):

    mu_gamma5 = mu_gamma_0 + beta_gamma*(logMstar - 11.4) + xi_gamma*(logRe - muR_logm(logMstar))

    res = normpdf(gamma5, mu_gamma5, sigma_gamma)
    return res

@njit 
def gamma5_generator(eta_gamma, logMstar, logRe, size:int):

    mu_gamma_0 ,beta_gamma,xi_gamma,sigma_gamma = eta_gamma

    mu_gamma5 = mu_gamma_0 + beta_gamma*(logMstar - 11.4) + xi_gamma*(logRe - muR_logm(logMstar))

    res = np.random.normal(loc = mu_gamma5, scale = sigma_gamma, size = size)

    return res



