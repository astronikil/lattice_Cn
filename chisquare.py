import numpy as np
from numpy.linalg import inv
from scipy import optimize
    
def func0(x):
    return x
    
class chiminimizer():
    """
    chisquare minimizing object.
    two methods currently: userdef or numpyopt. numpyopt works better.
    (note to myself: userdef is developed primarily to build in priors)
    """
    def __init__(self, 
                 method="numpyopt", 
                 func=func0, 
                 nvars=2, 
                 params0=[1.0,0.5],
                 isprior=False,
                 prior=None,
                ):
        self.method=method
        self.func=func
        self.nvars=nvars
        self.params0=params0
        self.isprior=isprior
        self.prior=prior
        if(self.isprior==True):
            self.nprior = len(self.prior)
    
    def chisq_userdef(self, 
                      params, 
                      func,
                      x, 
                      means, 
                      covary,
                      isinv=False
                     ):

        if(isinv):
            covary_inv = covary
        else:
            covary_inv = inv(covary)

        delta = func(x, *params)-means
        return np.dot(delta.T, np.dot(covary_inv, delta))

    def func_wapper_for_prior(self,
            data,
            *params,
            ):
        x_out_1 = self.func(data.T[:self.ndata].T, *params)
        x_out_2 = np.array(params)
        return np.append(x_out_1, x_out_2)

    def minimize(self, 
                 data
                ):
        x, means, covary = data

        if(self.isprior == True):
            len_data =  len(means)
            len_prior = self.nprior
            len_data_prior = len_data + len_prior
            self.ndata = len_data

            x_prior = np.zeros((2, len_prior))
            x_new = np.append(x, x_prior, axis=1)
            means_new = np.append(means, self.prior.T[0])
            func_new = self.func_wapper_for_prior

            cov_new = np.zeros((len_data_prior,len_data_prior))
            cov_new[0:len_data,0:len_data] = covary
            cov_new[len_data:len_data_prior, len_data:len_data_prior] = np.diag(self.prior.T[1]**2)
        else:
            x_new = x
            means_new = means
            func_new = self.func
            cov_new = covary

        if(self.method == "userdef"):
            covaryinv = inv(cov_new)
            res = optimize.minimize(self.chisq_userdef, self.params0,
                    args=(x_new, means_new, cov_new))
            chimin = self.chisq_userdef(res.x, x_new, means_new, covaryinv, isinv=True)
            return np.array([res.x, chimin])
        
        elif(self.method == "numpyopt"):
            
            res, covmat = optimize.curve_fit(
                func_new, x_new, means_new, p0=self.params0, sigma=cov_new, absolute_sigma=True,  maxfev=5000)
            chimin = self.chisq_userdef(res, func_new, x_new, means_new, cov_new) 
            self.minres = np.array([res, chimin])
            return self.minres
        
    def chiprofile(self, params, data, naxis):
         x, means, covary = data
         p0 = params[naxis]
         pmin = 0.8*p0
         pmax = 1.2*p0

         if(pmin > pmax):
             tmp = pmax
             pmax = pmin
             pmin = tmp

         pvec = np.linspace(pmin, pmax, 100)
         chivec = np.zeros(100)

         par = np.copy(params)
         chimin =  self.chisq_userdef( par, self.func, x, means, covary )

         i=0
         for p in pvec:
             par = np.copy(params)
             par[naxis] = p
             chivec[i] = self.chisq_userdef( par, self.func, x, means, covary ) - chimin
             i=i+1
         return pvec, chivec
