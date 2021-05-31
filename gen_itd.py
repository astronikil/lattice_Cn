from params import *
import numpy as np
from ope_basic import ope_basic

class gen_itd(ope_basic):
    
    def __init__(self,
                 latobj,
                 pdfobj,
                 correction_coeffs = [0.0, 0.0, 0.0],
                 L_qcd = 0.0,
                 ht_massdep = False,
                 c_latt = 0.0,
                 pa_massdep = False,
                 error_scale = 0.005,
                 ndata = 100
                ):
        
        self.latobj = latobj
        self.pdfobj = pdfobj
        self.moments = pdfobj.momlist()
        self.nterms = len(self.moments)-1
        self.hadname = pdfobj.hadname
        
        self.error_scale = error_scale
        
        if(self.hadname == 'proton'):
            self.hadmass = M_N
        elif(self.hadname == 'pion'):
            self.hadmass = M_pi
            
        if(ht_massdep):
            self.L_qcd = L_qcd*(self.hadmass/M_N)**1.0
        else:
            self.L_qcd = L_qcd
            
        if(pa_massdep):
            self.c_latt = c_latt*(M_N/self.hadmass)**1.0
        else:
            self.c_latt = c_latt
        
        ope_basic.__init__(self,
                     nterms = self.nterms,
                     correction_coeffs = correction_coeffs,
                     a_latt = latobj.a_latt,
                     mu = pdfobj.mu
                    )  
        self.ndata = ndata
        self.data = self.itd_syntheticdata_covmodel(ndata = ndata)

    def itd_centralval(self):
        #
        #   output central value of ITD
        #
        return self.twist2_nlo(self.latobj.znu, self.moments)\
            +  self.twist2_model(self.latobj.znu, self.moments)\
            +  self.higher_twist(self.latobj.znu)\
            +  self.lattice_artifact(self.latobj.znu)
    
    def higher_twist(self, znu):
        #
        # Higher twist correction modelled as an additive e^[- L^2 z^2 nu^2]-1.0
        #
        z, nu = self.latobj.znu
        return  np.exp( -(self.L_qcd * z * (self.a_latt / fac_mev_to_fm)  * nu)**2 ) - 1.0
    
    def lattice_artifact(self, znu):
        #
        # Lattice artifact is modelled as (a/z)^2 nu^2 = (P_z a)^2
        #
        z, nu = self.latobj.znu
        return self.c_latt * ( 1.0 / z )**2 * nu**2
        
    
    def itd_syntheticdata_covinput(self, cov, ndata=1, proj = 'real'):
        #
        #  output ndata random samples of the data with mean as itd_centralval
        #  and covariance matrix provided from outside
        #
        if(proj == 'real'):          
            mean = np.real(self.itd_centralval())
        else:
            mean = np.imag(self.itd_centralval())
        return np.random.multivariate_normal(mean, cov, ndata)
    
    def itd_syntheticdata_covmodel(self, ndata=100, proj='real'):
        #
        #  output ndata random samples of the data with mean as itd_centralval
        #  and covariance matrix from a built-in model in this class
        #
        cov = self.power_covariance(s=0.05, beta=1.0)
        return self.itd_syntheticdata_covinput(cov, ndata=ndata, proj=proj)
    
    def power_covariance(self, s=0.0, beta=0.5):
        #  An artificial covarance matrix whose off-diagonal entries decay 
        #  as 1/(nu-nu')^beta: cov(i,j) = er(i)*er(j)/(s*(nu(i)-nu(j)**beta+1))        
        z, nu = self.latobj.znu        
        ervec = np.abs(self.error_scale * nu * z)
        #ervec = self.error_scale*np.exp(nu * z)
        #print(len(z))
        #print(len(ervec))
        ndat = len(ervec)      
        ijlst = [[i, j] for i in range(ndat) for j in range(ndat)]
        return np.reshape(np.array( \
                [ervec[i]*ervec[j]*1.0/(s*np.abs((nu[i]-nu[j]))**beta+1.0) \
                 for i,j in ijlst ]), (ndat, ndat))
