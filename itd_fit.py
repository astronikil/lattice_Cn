import numpy as np
from ope_basic import ope_basic
from chisquare import *
from scipy.special import gamma as Gamma

class itd_fit():
    def __init__(self, 
                itdobj,
                iscov=False,
                method='numpyopt',
                outjck=False,
                fit_type='fit_moments',
                fit_nmom=30,
                fit_pa=False,
                fit_hightwist=False,
                param0=None,
                reorim='real',
                isprior=False,
                prior=None,
                kernel_type = 'nlo', # nlo, model, model+nlo, input
                wilson_input = lambda z, n: 0.0
                ):
        
        self.itdobj = itdobj
        self.hadname = itdobj.hadname
        self.pxmin = itdobj.latobj.pxmin
        self.pxmax = itdobj.latobj.pxmax
        self.method = method
        self.outjck = outjck
        self.fit_type = fit_type
        self.iscov = iscov
        self.isprior = isprior
        self.prior = prior
        self.fit_nmom = fit_nmom
        self.fit_pa = fit_pa
        self.fit_hightwist = fit_hightwist
        self.param0 = param0
        self.reorim = reorim
        self.opeobj = ope_basic(nterms = 2*self.fit_nmom)       
        self.itddata = self.itdobj.data #itd_syntheticdata_covmodel(ndata = 100)
        self.kernel_type = kernel_type
        self.wilson_input = wilson_input
        
        if(self.fit_pa == True and self.fit_hightwist == True):
            self.param0 = [ 1.e-6, 1.e-6 ]
            self.nvars = 2
            self.datakeys = ['ht', 'pa']
            self.func = self.ope_moments_wht_wpa
        elif(self.fit_pa == True and self.fit_hightwist == False):
            self.param0 = [ 1.e-3]
            self.nvars = 1
            self.datakeys = ['pa']
            self.func = self.ope_moments_wpa
        elif(self.fit_pa == False and self.fit_hightwist == True):
            self.param0 = [ 1.e-6]
            self.nvars = 1
            self.datakeys = ['ht']
            self.func = self.ope_moments_wht
        elif(self.fit_pa == False and self.fit_hightwist == False):
            self.param0 = []
            self.nvars = 0
            self.datakeys = []
            self.func = self.ope_moments           
        
        if(self.fit_type =='fit_moments'):
            self.momfunc = self.justmom
            self.nvars = self.nvars + self.fit_nmom
            self.param0.extend([ np.exp(-1.0*i) for i in range(1,self.fit_nmom+1) ])
            self.datakeys.extend([ 'x'+str(2*i-1) for i in range(1,self.fit_nmom+1) ])
            self.kern = self.ope_moments_kern
        
        if(self.fit_type == 'fit_coeffs'):            
            self.momlist = self.itdobj.moments
            self.nvars = self.nvars + self.fit_nmom
            self.param0.append([ [ 1.0 for zi in self.itdobj.latobj.z] for i in range(1,self.fit_nmom+1) ])          
            self.zfit = self.itdobj.latobj.z
            self.param0 =np.array(self.param0).ravel()
            self.datakeys.extend([ 'x'+str(2*i-1) for i in range(1,self.fit_nmom+1) ])
            self.kern = self.ope_coeff_kern

            self.prior = np.array([[ itdobj.cn_nlo(zi, 2*i), 1.0*itdobj.cn_nlo(zi, 2*i) ] for zi in self.itdobj.latobj.z for i in range(1,self.fit_nmom+1) ])
            self.isprior = True

        elif(self.fit_type =='fit_twoparams'):
            self.momfunc = self.mom_twoparam           
            self.nvars = self.nvars + 2
            self.param0.extend([ self.itdobj.pdfobj.alpha, self.itdobj.pdfobj.beta ])
            self.datakeys.extend([ 'alpha', 'beta' ])
            self.prior = np.array([ [self.itdobj.pdfobj.alpha, 2.0], [self.itdobj.pdfobj.beta, 2.0] ])
            self.isprior = True
            self.kern = self.ope_moments_kern
        
        if(kernel_type == 'nlo'):
            self.kernel_func = self.opeobj.twist2_nlo
        elif(kernel_type == 'nlo+model'):
            self.kernel_func = self.opeobj.twist2_nlo_model
        elif(kernel_type == 'model'):
            self.kernel_func = self.opeobj.twist2_model
        elif(kernel_type == 'input'):
            self.kernel_func = lambda z, n: self.opeobj.twist2_input_cn(z, n, wilson_input)
        elif(kernel_type == 'sample'):
            ff = lambda z, n: self.wilson_input(z, n, 0)
            self.kernel_func = lambda z, n: self.opeobj.twist2_input_cn(z, n, ff)
        
        self.fitter = chiminimizer(\
                    method = self.method, func=self.func, params0=self.param0,\
                    nvars=self.nvars, isprior=self.isprior, prior=self.prior)

    def justmom(self,
                n,
                *an):        
        if(self.reorim == 'real'):
            if(n%2==0):     
                if n//2-1 < 0:
                    return 1.0
                else:
                    return an[n//2-1]
            else:
                return 0.0
        elif(self.reorim == 'imag'):
            if(n%2 == 1):
                return an[(n-1)//2]
            else:
                return 0.0
            

    def coeff_linear_to_conventional(self,
                n,
                zi,
                *an):
        
        offset = np.int(zi)*self.fit_nmom
                
        if(self.reorim == 'real'):
            if(n%2==0):     
                if n//2-1 < 0:
                    return 1.0
                else:
                    return an[offset+n//2-1]
            else:
                return 0.0
        elif(self.reorim == 'imag'):
            if(n%2 == 1):
                return an[offset+(n-1)//2]
            else:
                return 0.0

    def mom_twoparam(self,
                    n,
                    a,
                    b):
        return (Gamma(2 + a + b)*Gamma(1 + a + n))/(Gamma(1 + a)*Gamma(2 + a + b + n))
    
    def ope_coeff_kern(self,
                      xin,
                      *amom
                      ):      
        z, nu = xin              
        xmom = np.array([self.momlist[n] for n in range(2*self.fit_nmom+1)])         
        coeffs = np.array([ [ self.coeff_linear_to_conventional(n, iz, *amom) for iz in range(len(self.zfit)) ]\
                           for n in range(2*self.fit_nmom+1)])
        ope_num =  self.opeobj.twist2_input_cn(xin, xmom, lambda zz, nn: coeffs[nn, np.int_(zz)-self.itdobj.latobj.zmin])
        return ope_num     

    def ope_moments_kern(self, 
                     xin, 
                     *amom
                     ):
        z, nu = xin         
        xmom = np.array([self.momfunc(n, *amom) for n in range(2*self.fit_nmom+1)]) 
        ope_num =  self.kernel_func(xin, xmom)
        return ope_num

    def ope_moments(self,
            xin,
            *amom):
        ope_num = self.kern(xin, *amom)
        return np.real(ope_num)

    def ope_moments_wpa(self,
            xin,
            ap,
            *amom):
        z, nu = xin
        p = nu/z
        ope_num = self.kern(xin, *amom)
        if(self.reorim=='real'):
            return ope_num + p*p*ap
        else:
            return ope_num + p*ap

    def ope_moments_wht(self,
            xin,
            ht,
            *amom):
        z, nu = xin        
        ope_num = self.kern(xin, *amom)
        if(self.reorim=='real'):
            return ope_num + z*z*nu*nu*ht
        else:
            return ope_num + z*z*nu*ht

    def ope_moments_wht_wpa(self,
            xin,
            ht,
            ap,
            *amom):
        z, nu = xin
        p = nu/z
        
        ope_num = self.kern(xin, *amom)
        if(self.reorim=='real'):
            return ope_num + z*z*ht*nu*nu + p*p*ap
        else:
            return ope_num + z*z*ht*nu + p*ap

    def fit(self, 
            fitrange = (1,2),
            verbose = False
           ):
        zmin = fitrange[0]; zmax = fitrange[1]  
        
        zbool = np.array([ True if (i <=zmax and  i>=zmin) else False for i in self.itdobj.latobj.z_extend])
        
        z_trunc = self.itdobj.latobj.z_extend[zbool]
        nu_trunc = self.itdobj.latobj.nu[zbool]                              
        zrange = np.array([z_trunc, nu_trunc])       
        itd_trunc = np.array([self.itddata[j][zbool] for j in range(len(self.itddata))])
        
        avg_trunc = np.mean(itd_trunc, axis=0) 
        er_trunc = np.std(itd_trunc, axis=0)        
        ndat = len(avg_trunc)
        if(self.iscov == True):
            cov_nrm = covar(itd_trunc)
            cov_trunc = np.array([[cov_nrm[i][j]*er_trunc[i]*er_trunc[j]
                  for j in range(ndat)] for i in range(ndat)])
        else:
            cov_trunc=np.diag(er_trunc**2) 
        
        iok=True
        icount=0
        while (iok & (icount<=10)):
            icount=icount+1
            try:                 
                self.fitter.params0, chi0 = self.fitter.minimize([zrange, avg_trunc, cov_trunc])
                iok=False
                if(verbose):
                    print('initial fit converged.')
            except ValueError:
                if(verbose):
                    print('initial fit did not converge at step', icount)

        if(icount==10):
            if(verbose):
                print('WARNING: fit guess did not work. Do not trust the final result.')

        ndof=np.shape(zrange)[1]-self.nvars
               
        def chisqwrapper(data):
            data_all = [ zrange, data, cov_trunc ]
            res, chimin = self.fitter.minimize(data_all)
            return res
        
        if(self.kernel_type == 'sample'):   
            self.resjck=np.zeros((len(itd_trunc), self.nvars))
            for j in range(len(itd_trunc)):
                ff = lambda z, n: self.wilson_input(z, n, j)
                self.kernel_func = lambda z, n: self.opeobj.twist2_input_cn(z, n, ff)
                self.resjck[j] = chisqwrapper(itd_trunc[j])                                                     
        else:      
            self.resjck = np.array([ chisqwrapper(itd_trunc[j]) for j in range(len(itd_trunc)) ])  
        
        mn = np.array([np.mean(self.resjck, axis=0), np.std(self.resjck, axis=0)])

        self.minres = {"fit": mn,
                "chimin":chi0, 
                "dof":ndof,
                "cdof":chi0/ndof, 
                "params0": self.fitter.params0,
                'zmin': zmin, 'zmax': zmax,
                'pmin': self.pxmin, 'pmax': self.pxmax, 
                 'zrange': np.arange(zmin, zmax+1) 
                  }

        if(verbose):
            return self.minres
        else:
            return 0
