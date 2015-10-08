import numpy as np
from sklearn.linear_model import SGDClassifier

class SGD2(SGDClassifier):
    
    augmentation = False

    def __init__(self, loss="hinge", penalty="l2", random_state=0, n_iter=5, shuffle=True, augmentation = False, alpha = 0.0001, l1_ratio=0.15):
        super(SGD2,self).__init__(loss=loss, penalty=penalty,random_state=random_state,n_iter=n_iter,shuffle=shuffle, alpha = alpha, l1_ratio=l1_ratio)
        self.augmentation = augmentation
    
    def partial_fit(self, X, y):
        if self.augmentation:
            y = np.repeat(y, 10)
            
        return super(SGD2,self).partial_fit(X,y)
    
    def fit(self, X, y):
        if self.augmentation:
            y = np.repeat(y, 10)
            
        return super(SGD2,self).fit(X,y)
    
    def predict(self, X):
        y_pred = super(SGD2,self).predict(X)
        
        if self.augmentation:
            y_pred =np.around(np.mean(y_pred.reshape((-1,10)),axis=1))
             
        return y_pred
    