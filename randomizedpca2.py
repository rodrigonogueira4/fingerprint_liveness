import numpy as np
from sklearn.decomposition import RandomizedPCA

class RandomizedPCA2(RandomizedPCA):
    
    def __init__(self, n_components, whiten=True,random_state=0, copy=False):
        super(RandomizedPCA2,self).__init__(n_components=n_components, whiten=whiten,random_state=random_state, copy=copy )
    
    def fit(self, X, y):
        
        return super(RandomizedPCA2,self).fit(X,y)
    
    def transform(self, X):
          
        return super(RandomizedPCA2,self).transform(X).astype(np.float32)
    
    def fit_transform(self, X, y):
        
        return super(RandomizedPCA2,self).fit_transform(X,y).astype(np.float32)
    
    