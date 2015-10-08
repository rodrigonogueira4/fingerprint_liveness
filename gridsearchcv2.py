from sklearn.grid_search import ParameterGrid
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from datetime import datetime
#from multiprocessing import Pool
import multiprocessing.pool # We must import this explicitly, it is not imported by the top-level multiprocessing module.
import copy
import sys

class GridSearchCV2():
    
    verbose = 1
    cv = None
    param_grid = {}
    param_grid_new = {}
    estimators = None
    auto_adjust_params = None
    best_score_ =None
    best_params_ =None
    refit = True
    augmentation = False
    testing = None    
    def __init__(self, estimators, param_grid, cv, verbose=0, n_jobs=1, n_jobs_last_estimator=1, refit = False, augmentation = False, auto_adjust_params = None, testing=None):
        self.verbose = verbose
        self.cv = cv
        self.param_grid = param_grid
        self.estimators = estimators
        self.auto_adjust_params = auto_adjust_params
        self.n_jobs = n_jobs
        self.n_jobs_last_estimator = n_jobs_last_estimator
        self.refit = refit
        self.augmentation = augmentation
        self.testing = testing
        
    def fit(self, X, y):
    
        if self.verbose >=1:
            print 'Running GridSearchCV...'
                
        #p = Pool(processes=self.n_jobs)
        p = MyPool(processes=self.n_jobs)
        
        #Get parameters for multiprocessing task:
        #get parameters that belongs only to first estimator. The remaining belongs to the other estimators
        param_grid_mine, param_grid_others = self.split_params(self.param_grid, self.estimators.steps[0][0])
        params_vars = list(ParameterGrid(param_grid_mine)) #vary all parameters for this estimator only
        
        if False: #set to TRUE to make debug of inner functions easier. It does not use multi-task in this case
            print 'vai dar erro, remover isso daki!'
            lst_scores, lst_params = self.est_var(self.estimators.steps, self.param_grid, X, y,params_vars=params_vars)
            
        params_pool =[]
    
        #divide the first estimator parameters among n_jobs
        for i in range(self.n_jobs):
            params_pool.append(params_vars[i::self.n_jobs])
            
        #for multiple arguments and python 3.3, use pool.starmap()        
        j=len(params_pool)
        results = p.map(unwrap_self_est_var, zip([self]*j, [self.estimators.steps]*j, [param_grid_others]*j, [X]*j, [y]*j, [None]*j, [None]*j, params_pool))
        
        p.close() #terminate process
        p.join()
        
        lst_scores = []
        lst_params = []
        for result in results:
            lst_scores.extend(result[0])
            lst_params.extend(result[1])
        
        #get the best score and best parameters
        best_score = np.asarray(lst_scores).min()
        idx_best = np.where(lst_scores == best_score)
        best_params = [lst_params[i] for i in idx_best[0]]
        
        self.best_score_ =best_score 
        self.best_params_ = best_params
        
        if self.refit:
            # fit the best estimator using the entire dataset
            self.set_params(self.best_params_[0])
            self.estimators= self.estimators.fit(X, y)
            
        if self.verbose>=1:
            print 'best_params = '
            for best_param in best_params:
                print best_param+'\n'
    
        return self
    
    
    #set params for all estimators
    def set_params(self, params):
        for name, estimator in self.estimators.steps:
            #get only the params for the current estimator
            estimator =self.set_estimator_params(name, estimator, params)
            
    #set params for a specific estimator
    def set_estimator_params(self, estimator_name, estimator, param_var):
        
        #first remove the estimator name from the name value
        for key, value in param_var.items():
            if key[:key.find('__')] == estimator_name:
                estimator.set_params(**{key[key.find('__')+2:]: value})
        return estimator
     
    def predict(self, X):
        
        return self.estimators.predict(X)
        
    def split_params(self, param_grid, estimator_name):
        
        param_grid_mine = {}
        param_grid_others = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            if key[:key.find('__')] == estimator_name:
                param_grid_mine[key] =copy.deepcopy(value)
                del param_grid_others[key]
        
        return param_grid_mine, param_grid_others
    
    
    def est_var(self, estimators, param_grid, X_train, y_train, X_test=None, y_test=None, params_vars=None):
        
        curr_estimator = estimators[0][1]
        curr_estimator_name = estimators[0][0]
         
        lst_params=[]
        scores_var2 = []
        
        param_grid_mine, param_grid_others = self.split_params(param_grid, curr_estimator_name)
        if params_vars==None: #workaround for passing the divided parameters when multiple jobs are running 
            params_vars = list(ParameterGrid(param_grid_mine)) #vary all parameters for this estimator only
        
        for param_var in params_vars:
            
            antes1 = datetime.now()
            
            curr_estimator = self.set_estimator_params(curr_estimator_name, curr_estimator,param_var)
            
            if hasattr(curr_estimator, 'fit_transform') and not len(estimators) ==1: #check if the estimator has the transform method and it is not the classifier
                X_train_t = curr_estimator.fit_transform(X_train,y_train)
                
                if X_test!=None: #if the test dataset was passed, transform it
                    X_test_t = curr_estimator.transform(X_test)
            elif hasattr(curr_estimator, 'transform') and not len(estimators) ==1: #check if the estimator has the transform method
                curr_estimator = curr_estimator.fit(X_train,y_train)
                X_train_t = curr_estimator.transform(X_train)
            
                if X_test!=None: #if the test dataset was passed, transform it
                    X_test_t = curr_estimator.transform(X_test)
                    
            else: #if it doesn't have the transform method, just pass the original dataset ahead
                curr_estimator = curr_estimator.fit(X_train,y_train)
                X_train_t = X_train
                X_test_t = X_test
            
            if self.verbose>=2:
                print 'time transform', curr_estimator_name, datetime.now() - antes1
                sys.stdout.flush()#force print when running child/sub processes
            
            Xy_masks = []
            if self.estimators.steps[0][0] == curr_estimator_name: #if this is the first estimator, make cross validation split
                if not isinstance(self.cv, list): #case normal k-folds
                    kf = StratifiedKFold(y_train, n_folds=self.cv, indices=False)
                    for train_mask, test_mask in kf:
                        if self.augmentation:
                            Xy_masks.append([X_train_t[train_mask,:], X_train_t[test_mask,:], y_train[train_mask[::10]], y_train[test_mask[::10]]]) # split training using masks
                        else:
                            Xy_masks.append([X_train_t[train_mask,:], X_train_t[test_mask,:], y_train[train_mask], y_train[test_mask]]) # split training using masks
                else:  #case n x k folds
                    for i in range(self.cv[0]):
                        nsize = X_train_t.shape[0]
                        isize = nsize/self.cv[0]
                        jsize = nsize/self.cv[1]
                        for j in range(self.cv[1]):
                            train_mask = np.zeros((nsize), dtype=np.bool)
                            train_true_idx = np.mod(np.arange(i*isize+j*jsize,i*isize+j*jsize+jsize),nsize)
                            train_mask[train_true_idx] = True
                            test_mask = np.invert(train_mask)
                            if self.augmentation:
                                Xy_masks.append([X_train_t[train_mask,:], X_train_t[test_mask,:], y_train[train_mask[::10]], y_train[test_mask[::10]]]) # split training using masks
                            else:
                                Xy_masks.append([X_train_t[train_mask,:], X_train_t[test_mask,:], y_train[train_mask], y_train[test_mask]]) # split training using masks
                
            else: #if this is not the first estimator, just use normal/given train and test sets
                Xy_masks.append([X_train_t,X_test_t,y_train,y_test])
            
            scores_var = []
            weights = []
            for X_train_u, X_test_u, y_train_u, y_test_u in Xy_masks:
                weights.append(y_test_u.shape[0])
                if len(estimators) ==1: #if this is the last estimator in the chain, predict and report back the score
                    """
                    try:
                        y_pred = curr_estimator.decision_function(X_test_u).ravel()
                    except (NotImplementedError, AttributeError):
                        y_pred = curr_estimator.predict_proba(X_test_u)[:, 1]
                        y_pred[np.isnan(y_pred)] = False #VERIFICAR AQUI!!!
                    """
                    
                    y_pred = curr_estimator.predict(X_test_u)
                    
                    scores = [100.-(100.*roc_auc_score(y_test_u, y_pred))]
                    params = [param_var]
                
                
                #MUDAR AQUI DEPOIS PARA FICAR MAIS GENERICO
                elif False: #len(estimators) ==2: #if the next estimator is the last, use multi processing
                        
                    params_vars_temp = list(ParameterGrid(param_grid_others)) #varies all parameters for the next estimator
                    p = MyPool(processes=self.n_jobs_last_estimator)
                    
                    params_pool =[]
                
                    #divide the first estimator parameters among n_jobs
                    for i in range(self.n_jobs_last_estimator):
                        params_pool.append(params_vars_temp[i::self.n_jobs_last_estimator])
                        
                    #for multiple arguments and python 3.3, use pool.starmap()        
                    j=len(params_pool)
                    results = p.map(unwrap_self_est_var, zip([self]*j, [estimators[1:]]*j, [param_grid_others]*j, [X_train_u]*j, [y_train_u]*j, [X_test_u]*j, [y_test_u]*j, params_pool))
                    p.close() #terminate processes
                    p.join()
                    
                    scores = []
                    params = []
                    for result in results:
                        scores.extend(result[0])
                        params.extend(result[1])
                else:
                    scores, params = self.est_var(estimators[1:], param_grid_others, X_train_u, y_train_u, X_test_u, y_test_u )
                    
                for param in params: #add the current parameter to all configurations
                    param.update(param_var)
            
                scores_var.append(scores)
            
            scores_var = np.asarray(scores_var).reshape(len(Xy_masks),-1)
            scores_avg = np.average(scores_var, axis=0, weights=np.asarray(weights))
            scores_std = np.std(scores_var, axis=0)
            scores_var2.extend(scores_avg)
            
            lst_params.extend(params)
        
            if self.estimators.steps[0][0] == curr_estimator_name: #if this is the first estimator,
                totaltime = datetime.now() - antes1
                for i in range(len(params)):
                    self.testing.append_results(params=params[i], score_mean=scores_avg[i], score_std=scores_std[i], total_time=totaltime/len(params))
                
                if self.verbose>=2:
                    print 'param_var=', param_var, 'time=', totaltime
                                                    
        return scores_var2, lst_params
    
def unwrap_self_est_var(arg, **kwarg):
    return GridSearchCV2.est_var(*arg, **kwarg)


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

