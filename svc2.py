import numpy as np
from sklearn.svm import SVC

class SVC2(SVC):

    def __init__(self, kernel, class_weight, C, gamma, random_state=0, multi_column = False, augmentation = False, aug_rotate = False):
        super(SVC2,self).__init__(kernel=kernel, class_weight=class_weight,random_state=random_state,C=C, gamma=gamma)
        self.augmentation = augmentation
        self.aug_rotate = aug_rotate
        self.multi_column = multi_column
        if self.augmentation or self.multi_column:
            if aug_rotate:
                self.multiply = 30
            else:
                self.multiply = 10
        
            
    def fit(self, X, y):
        if self.augmentation:
            y = np.repeat(y, self.multiply )
                
        return super(SVC2,self).fit(X,y)
    
    def predict(self, X):
        if self.multi_column:
            y_pred = super(SVC2,self).decision_function(X)
        else:    
            y_pred = super(SVC2,self).predict(X)
            if self.augmentation:
                y_pred =np.around(np.mean(y_pred.reshape((-1,self.multiply )),axis=1))
          
                
        return y_pred
    