import numpy as np
from scipy.special import gamma

class pdf():
    def __init__(self,
            alpha=-0.5,
            beta=2.0, 
            s=0.0,
            t=0.0,
            mu=3.2,
            nmom = 30,
            hadname = 'noname'):
        self.alpha = alpha
        self.beta = beta
        self.s = s
        self.t = t
        self.mu = mu
        self.nmom = nmom
        self.nlst = np.int_(np.linspace(0, nmom, nmom+1))
        self.norm = gamma(1 + beta)*(\
                (s*gamma(1.5 + alpha))/gamma(2.5 + alpha + beta)\
                + ((2 + alpha + beta + t + alpha*t)*gamma(1 + alpha))/gamma(3 + alpha + beta)\
                )
        self.hadname = hadname

    def f(self,
         x):
        return x**self.alpha*(1.0-x)**self.beta*(1.0+s*np.sqrt(x)+t*x)/self.norm
    
    def mom(self,
           n):
        return ((self.s*gamma(1.5 + self.alpha + n))/gamma(2.5 + self.alpha + self.beta + n) + ((2 + self.alpha + self.beta + n + (1 + self.alpha + n)*self.t)*gamma(1 + self.alpha + n))/gamma(3 + self.alpha + self.beta + n))/\
               ((self.s*gamma(1.5 + self.alpha))/gamma(2.5 + self.alpha + self.beta) + ((2 + self.alpha + self.beta + self.t + self.alpha*self.t)*gamma(1 + self.alpha))/gamma(3 + self.alpha + self.beta))

    def momlist(self):
        return self.mom(self.nlst)
