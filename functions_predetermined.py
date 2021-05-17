import numpy as np

class functions_predetermined():
    def __init__(self):
        self.harmonic_1 = self.harmonicfunc_1(nmax=100)
        self.harmonic_2 = self.harmonicfunc_2(nmax=100)
        self.euler = 0.57721566490153286060651209008240243

    def harmonicfunc_1(self, nmax=100):
        # H(n) = sum_i 1/i
        lst = np.zeros(nmax+1)
        s=0
        lst[0] = 0.0
        for i in range(1,nmax+1):
            s=s+1.0/i
            lst[i]=s
        return lst
    
    def harmonicfunc_2(self, nmax=100):
        # H_2(n) = sum_i 1/i^2
        lst = np.zeros(nmax+1)
        s=0
        lst[0] = 0.0
        for i in range(1,nmax+1):
            s=s+1.0/i**2
            lst[i]=s
        return lst
    

