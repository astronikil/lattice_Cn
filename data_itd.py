from params import *
import numpy as np
from gen_itd import gen_itd

class data_itd(gen_itd):
    
    def __init__(self,
                 latobj,
                 pdfobj,
                 datafile,
                 ndata = 100
                ):
        
        self.latdata = np.loadtxt(datafile)

        gen_itd.__init__(self, 
                 latobj,
                 pdfobj,
                 correction_coeffs = [0.0, 0.0, 0.0],
                 L_qcd = 0.0,
                 ht_massdep = False,
                 c_latt = 0.0,
                 pa_massdep = False,
                 error_scale = 1.0,
                 ndata = ndata)

        self.data = self.itd_syntheticdata_covmodel(ndata = ndata)


    def itd_centralval(self):
        #
        #   output central value of ITD
        #
        return self.latdata.T[2]
    
    def power_covariance(self, s=0.0, beta=0.5):
        #  An artificial covarance matrix whose off-diagonal entries decay 
        #  as 1/(nu-nu')^beta: cov(i,j) = er(i)*er(j)/(s*(nu(i)-nu(j)**beta+1))        
        z, nu = self.latobj.znu        
        ervec = np.abs(self.error_scale * self.latdata.T[3])
        print('foo', len(ervec))
        ndat = len(ervec)      
        ijlst = [[i, j] for i in range(ndat) for j in range(ndat)]
        return np.reshape(np.array( \
                [ervec[i]*ervec[j]*1.0/(s*np.abs((nu[i]-nu[j]))**beta+1.0) \
                 for i,j in ijlst ]), (ndat, ndat))
