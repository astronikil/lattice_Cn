from params import *
import numpy as np
from wilson_coeff import wilson_coeff
from scipy.special import factorial

class ope_basic(wilson_coeff):
    def __init__(self,
                 nterms = 20,
                 correction_coeffs = [0.0, 0.0, 0.0],
                 a_latt = 0.06,
                 mu = 3.2
                ):
        
        wilson_coeff.__init__(self,
                     nterms = nterms,
                     correction_coeffs = correction_coeffs,
                      )
        self.a_latt = a_latt 
        self.nbasis = np.int_(np.linspace(0,self.nterms,self.nterms+1))
        self.taylorbasis = (-1.0j)**self.nbasis/factorial(self.nbasis)
        self.mu = mu
    
    def cn(self, z, n, coeffs):
        #
        # sum up all the log(z) terms to construct the full coefficient c_n(z)
        #
        terms = [ coeffs[n, i]*np.log( (self.mu*z*self.a_latt/fac_mev_to_fm)**2 )**i for i in range(self.nlogs) ]
        return np.sum(terms, axis=0)
    
    def cn_nlo(self, z, n):
        #
        # sum up all the log(z) terms to construct the full coefficient c_n(z)
        #
        return self.cn(z, n, self.c_pert)
    
    def cn_model(self, z, n):
        #
        # sum up all the log(z) terms to construct the full coefficient c_n(z)
        #
        return self.cn(z, n, self.c_model)
    
    def cn_nlo_plus_model(self, z, n):
        #
        # sum up all the log(z) terms to construct the full coefficient c_n(z)
        #
        return self.cn(z, n, self.c_pert)+self.cn(z, n, self.c_model)
       
    def twist2_input_cn(self,
                    znu,
                    moments,
                    coeffs):
        #
        #  construct the twist-2 OPE at fixed z, nu with z-dependent wilson coeffs 
        #  that are provided as an array input
        #
        z, nu = znu
        tmp = [ self.taylorbasis[n]*moments[n]*coeffs(z, n)*nu**n for n in self.nbasis ]
        return np.sum(tmp, axis=0)
    
    def twist2(self, 
               znu,
               moments,
               coeffs):
        #
        #  construct the twist-2 OPE at fixed z, nu with wilson coeffs given by coeffs
        #
        return self.twist2_input_cn(znu, moments, lambda zz, nn: self.cn(zz, nn, coeffs))
    
    def twist2_nlo(self,
                  znu,
                  moments):
        #
        # twist-2 ope using NLO wilson coeffs
        #  
        return self.twist2(znu, moments, self.c_pert)
    
    def twist2_model(self,
                  znu,
                  moments):
        #
        # twist-2 ope using user-defined wilson coeffs (or corrections to NLO terms)
        #  
        return self.twist2(znu, moments, self.c_model)
    
    def twist2_nlo_model(self,
                  znu,
                  moments):
        #
        # twist-2 ope using user-defined wilson coeffs (or corrections to NLO terms)
        #  
        return self.twist2(znu, moments, self.c_pert)+self.twist2(znu, moments, self.c_model) 

#opeobj = ope_basic(nterms=3)
#opeobj.twist2_nlo(np.array([[1, 2, 5, 6], [2.3, 4.5, 7.8, 9.1]]), np.array([1.0, 0.25, 0.06, 0.06]))
