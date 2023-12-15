#region imports
from collections import deque
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import kendalltau
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy import stats
import numpy as np

import sys

#endregion

def set_parameter(family, tau):
    ''' Estimate the parameters for three kinds of Archimedean copulas
    according to association between Archimedean copulas and the Kendall rank correlation measure
    '''

    if  family == 'clayton':
        return 2 * tau / (1 - tau)
    
    elif family == 'frank':
        
        '''
        debye = quad(integrand, sys.float_info.epsilon, theta)[0]/theta  is first order Debye function
        frank_fun is the squared difference
        Minimize the frank_fun would give the parameter theta for the frank copula 
        ''' 
        integrand = lambda t: t / (np.exp(t) - 1)  # generate the integrand
        frank_fun = lambda theta: ((tau - 1) / 4.0  - (quad(integrand, sys.float_info.epsilon, theta)[0] / theta - 1) / theta) ** 2
        
        return minimize(frank_fun, 4, method='BFGS', tol=1e-5).x[0]
    
    elif family == 'gumbel':
        return 1 / (1 - tau)

def lpdf_copula(family, theta, u, v):
    '''Estimate the log probability density function of three kinds of Archimedean copulas
    '''

    if family == 'clayton':
        pdf = (theta + 1) * ((u ** (-theta) + v ** (-theta) - 1) ** (-2 - 1 / theta)) * (u ** (-theta - 1) * v ** (-theta - 1))
        
    elif family == 'frank':
        num = -theta * (np.exp(-theta) - 1) * (np.exp(-theta * (u + v)))
        denom = ((np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) + (np.exp(-theta) - 1)) ** 2
        pdf = num / denom
        
    elif family == 'gumbel':
        A = (-np.log(u)) ** theta + (-np.log(v)) ** theta
        c = np.exp(-A ** (1 / theta))
        pdf = c * (u * v) ** (-1) * (A ** (-2 + 2 / theta)) * ((np.log(u) * np.log(v)) ** (theta - 1)) * (1 + (theta - 1) * A ** (-1 / theta))
        
    return np.log(pdf)

def misprice_index(window_list,pair_list,theta,copula,ecdf_x,ecdf_y):
    '''Calculate mispricing index for every day in the trading period by using estimated copula
    Mispricing indices are the conditional probability P(U < u | V = v) and P(V < v | U = u)'''        
    return_x = np.log(window_list[pair_list[0]][-1] / window_list[pair_list[0]][-2])
    return_y = np.log(window_list[pair_list[1]][-1] / window_list[pair_list[1]][-2])
    
    # Convert the two returns to uniform values u and v using the empirical distribution functions
    u = ecdf_x(return_x)
    v = ecdf_y(return_y)
    
    if copula == 'clayton':
        MI_u_v = v ** (-theta - 1) * (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta - 1) # P(U<u|V=v)
        MI_v_u = u ** (-theta - 1) * (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta - 1) # P(V<v|U=u)

    elif copula == 'frank':
        A = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) + (np.exp(-theta * v) - 1)
        B = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) + (np.exp(-theta * u) - 1)
        C = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) + (np.exp(-theta) - 1)
        MI_u_v = B / C
        MI_v_u = A / C
    
    elif copula == 'gumbel':
        A = (-np.log(u)) ** theta + (-np.log(v)) ** theta
        C_uv = np.exp(-A ** (1 / theta))   # C_uv is gumbel copula function C(u,v)
        MI_u_v = C_uv * (A ** ((1 - theta) / theta)) * (-np.log(v)) ** (theta - 1) * (1.0 / v)
        MI_v_u = C_uv * (A ** ((1 - theta) / theta)) * (-np.log(u)) ** (theta - 1) * (1.0 / u)

    return MI_u_v, MI_v_u