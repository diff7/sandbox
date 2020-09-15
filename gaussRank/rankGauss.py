import numpy as np
from scipy.special import erfinv
from scipy.stats import rankdata


class GRank:
    def __init__(self):
        self.half_range = 0.99
        self.range = self.half_range*2
        self.mapper = {}

    def fit(self, X):
        self.min = X.min() 
        self.max = X.max() 
        rank =  rankdata(X) 
        scale =  self.range/len(rank)
        
    def transform(self,X):
        X = np.where(X <= self.min, self.min, X)
        X = np.where(X >= self.max, self.max, X)
        rank =  rankdata(X) 
        scale =  self.range/len(rank)
        rank = rank*scale - self.half_range
        return erfinv(rank)
    
class GpandasRanker:
    def __init__(self):
        self.rankers = {}

    def fit(self, d_set, columns):
        for col in columns:
            X = d_set[col].values
            ranker = GRank()
            ranker.fit(X)
            self.rankers[col] = ranker

    def transform(self, d_set):
        d_set = d_set.copy()
        for col in self.rankers:
            d_set[col] = self.rankers[col].transform(d_set[col].values)
        return d_set