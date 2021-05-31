from params import *
import numpy as np
from functions_predetermined import functions_predetermined

class wilson_coeff(functions_predetermined):
    def __init__(self,
                 nterms = 20,
                 correction_coeffs = [0.0, 0.0, 0.0]
                 #al_s = 0.1,        
                ):
        
        functions_predetermined.__init__(self)
        self.nterms = nterms
        self.al_s = alpha_s*C_F/tpi
        self.correction_coeffs = np.array(correction_coeffs)
        self.nlogs = 3 #hard coded
        
        self.c_pert = np.zeros((self.nterms+1, self.nlogs))
        self.c_model = np.zeros((self.nterms+1, self.nlogs))
        for n in range(nterms+1):
            self.c_pert[n] = self.c_nlo(n)
            self.c_model[n] = self.c_artificial(n)
        
    def c_nlo(self,
             n):
        #
        #  output NLO coeffs of log(z)^0, log(z)^1, log(z)^2 as a tuple
        #
        return np.array([ self.c_nlo_log0(n), self.c_nlo_log1(n), self.c_nlo_log2(n)])
    
    def c_artificial(self, n):
        #
        #  output user-defined corrections to coefficients to log(z)^0, log(z)^1, log(z)^2 terms as a tuple
        #
        return self.correction_coeffs * np.array([ self.c_artificial_log0(n), self.c_artificial_log1(n), self.c_artificial_log2(n)])
        
    def c_nlo_log0(self, n):
        #
        # coefficient of log(z)^0 at NLO
        #
        return 1.0+self.al_s*(((3.0+2.0*n)/(2.0+3.0*n+n**2)+2.0*self.harmonic_1[n])*np.log(np.exp(2.0*self.euler)/4.0)\
                                +(5.0+2.0*n)/(2.0+3.0*n+n**2)\
                                +2.0*(1.0-self.harmonic_1[n])*self.harmonic_1[n]-2.0*self.harmonic_2[n]\
                                -1.5*np.log(np.exp(2.0*self.euler)/4.0)-2.5)
    def c_nlo_log1(self, n):
        #
        # coefficient of log(z)^1 in c_n at NLO
        #
        return self.al_s*((3.0+2.0*n)/(2.0+3.0*n+n**2)+2.0*self.harmonic_1[n] - 1.5)

    def c_nlo_log2(self, n):
        #
        # coefficient of log(z)^2 in c_n at NLO
        #
        return 0.0
    
    def c_artificial_log0(self, n):
        #
        # user-defined correction to log(z)^0 c_n term
        #
        #return (self.al_s)**2*self.harmonic_1[n] #np.log(n+1)
        return (self.al_s)**2*np.log(n+1)

    def c_artificial_log1(self, n):
        #
        # user-defined correction to log(z)^1 c_n term
        #
        #return (self.al_s)**2*self.harmonic_1[n] #np.log(n+1)
        return (self.al_s)**2*np.log(n+1)

    def c_artificial_log2(self, n):
        #
        # user-defined correction to log(z)^2 c_n term
        #
        #return (self.al_s)**2*self.harmonic_1[n] #np.log(n+1)
        return (self.al_s)**2*np.log(n+1)

#wobj = wilson_coeff(nterms = 20, correction_coeffs = [0.1, 0.05, 0.01],  )
#wobj.c_model 
