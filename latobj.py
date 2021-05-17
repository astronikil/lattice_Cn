import numpy as np

class latobj():
    def __init__(self,
                Lz = 48,
                a_latt = 0.06,
                zmin = 1,
                zmax = 12,
                pmin = 1,
                pmax = 5):
        self.Lz = Lz
        self.a_latt = a_latt
        
        self.pxmin = pmin
        self.pxmax = pmax
        self.zmin = zmin
        self.zmax = zmax
        
        self.z = np.int_(np.linspace(zmin, zmax, zmax-zmin+1)) 
        self.nzvals = len(self.z)   
        
        self.pz = np.linspace(pmin, pmax, pmax-pmin+1)*2.0*np.pi/self.Lz       
        self.npzvals = len(self.pz)     
        
        self.nu = np.outer(self.pz,self.z).flatten()        
        self.nuvals = len(self.nu) 
        
        self.z_extend = np.int_(np.resize(self.z, self.nuvals))       
        self.znu = np.array([self.z_extend, self.nu])
