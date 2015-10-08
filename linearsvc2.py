import numpy as np
from sklearn.svm import LinearSVC

class LinearSVC2(LinearSVC):
    
    augmentation = False

    def __init__(self, C, fit_intercept=True, random_state=0, class_weight='auto', augmentation = False):
        super(LinearSVC2,self).__init__(random_state=random_state,fit_intercept=fit_intercept, class_weight=class_weight, C=C)
        self.augmentation = augmentation
    
    def fit(self, X, y):
        if self.augmentation:
            y = np.repeat(y, 10)
            
        return super(LinearSVC2,self).fit(X,y)
    
    def fit_transform(self, X, y):
        if self.augmentation:
            y = np.repeat(y, 10)
            
        return super(LinearSVC2,self).fit_transform(X,y)
    
    def predict(self, X):
        y_pred = super(LinearSVC2,self).predict(X)
        
        if self.augmentation:
            y_pred =np.around(np.mean(y_pred.reshape((-1,10)),axis=1))
             
        return y_pred
    