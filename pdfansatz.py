import numpy as np
from scipy.special import gamma

class pdf():
    def __init__(self,
            alpha=-0.5,
            beta=2.0, 
            mu=3.2,
            nmom = 30,
            hadname = 'noname'):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nmom = nmom
        self.nlst = np.int_(np.linspace(0, nmom, nmom+1))
        self.norm = (gamma(1 + alpha)*gamma(1 + beta))/gamma(2 + alpha + beta)
        self.hadname = hadname
        
    def f(self,
         x):
        return x**self.alpha*(1.0-x)**self.beta/self.norm
    
    def mom(self,
           n):
        return (gamma(2 + self.alpha + self.beta)*gamma(1 + self.alpha + n))/\
        (gamma(1 + self.alpha)*gamma(2 + self.alpha + self.beta + n))

    def momlist(self):
        return self.mom(self.nlst)
