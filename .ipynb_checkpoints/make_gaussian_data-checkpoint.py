import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class Gauz:
    def __init__(self, 
                 num_dists,
                 size):
        self.num_dists = num_dists
        self.size = int(size / num_dists)
        self.dists = []
        self.max_random = 200
        self.make_distributions()
        self.color = 'green'
        
    def make_distributions(self):
        class Dist:
            def __init__(self):
                self.m = 0
                self.sigme = 0
                self.vars = []
        
        variables = []        
        for i in range(self.num_dists):
            dist = Dist()
            dist.m = np.random.randint(0,self.max_random)
            dist.sigma = np.random.randint(0,self.max_random)
            dist.vars = np.random.normal(dist.m, np.sqrt(dist.sigma), self.size)
            variables+=list(dist.vars)
            self.dists.append(dist)

        self.variables = np.array(variables)
        self.bins = np.linspace(self.variables.min(), self.variables.max(), int(self.size*self.num_dists))
        np.random.shuffle(self.variables)
    
    def pdf(self, data, dist_idx):
        if len(self.dists) < dist_idx:
            print(f'No such distribution, max distributions : {len(self.dists)}')
            return None
        data =np.array(data)
        d = self.dists[dist_idx]
        
        s1 = 1/(np.sqrt(2*np.pi*d.sigma))
        s2 = np.exp(-(np.square(data - d.m)/(2*d.sigma)))
        
        return s1 * s2
    
    def plot(self):

        plt.figure(figsize=(10,7))
        plt.xlabel("$x$")
        plt.ylabel("pdf")
        plt.scatter(self.variables, [0.005]*len(self.variables), color='navy', s=30, marker=2, label="Train data")
        
        label = "True pdf"
        for idx in range(self.num_dists):
            plt.plot(self.bins, self.pdf(self.bins, idx), color=self.color, label=label)
            label = None
        
        plt.legend()
        plt.plot()