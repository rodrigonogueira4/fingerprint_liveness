from testing import Testing

if __name__ == '__main__':
    """
    testing = Testing()
    testing.divide_by = 1
    testing.cross_validation = True
    testing.n_folds = [5,2]
    testing.augmentation = False
    testing.predict = 'SVM'
    testing.n_processes_cv = 1
    testing.n_processes_cv_last_estimator = 1
    testing.n_processes_pproc =20
    testing.use_pca= True
    testing.use_lda= False
    testing.svm__kernel = 'rbf'
    testing.params_pproc['pproc__feature_extractor_name'] = ['LBP']
    testing.params_pproc['pproc__size_percentage'] = [0.5, 1.0]
    #testing.params_pproc['pproc__size_percentage'] = [1.0]
    testing.params_lbp['pproc__feature_extractor__method'] = ['uniform', 'default']
    #testing.params_lbp['pproc__feature_extractor__method'] = ['uniform']
    testing.params_lbp['pproc__feature_extractor__n_tiles'] = [[1,1],[3,3],[5,5],[7,7]]
    #testing.params_lbp['pproc__feature_extractor__n_tiles'] = [[1,1],[7,7]]
    testing.params_auto['pca__n_components'] = [10, 30, 50, 100, 300, 500]
    #testing.params_auto['pca__n_components'] = [100]
    testing.params_svm['pred__C'] = [0.0001, 0.01, 0.01, 1, 100, 5000]
    #testing.params_svm['pred__C'] = [1000]
    testing.params_svm['pred__gamma'] = [0.0001, 0.001, 0.01, 0.1]
    #testing.params_svm['pred__gamma'] = [0.001]
    testing.var_sensor('LBP',datasettrain='all',sensortrain ='all',datasettest='all',sensortest ='all')
    #testing.var_sensor('LBP','all','')
    """
    
    testing = Testing()
    testing.size_percentage = 1.0
    testing.divide_by = 80
    testing.cross_validation = False
    testing.augmentation = False
    testing.aug_rotate = False
    testing.multi_column = False       
    testing.roi = False
    testing.low_pass = False
    testing.high_pass = False
    testing.n_processes_pproc = 3
    testing.n_processes_cv =1
    testing.lbp__method = 'uniform'
    testing.lbp__n_tiles = [1,1]
    testing.predict = 'SVM'
    testing.svm__kernel='rbf'
    testing.svm__gamma = 0.01
    testing.svm__C = 1000
    #testing.predict = 'SGD'
    #testing.sgd__alpha = 0.0001
    testing.use_pca = True
    testing.pca__n_components = 100
    #testing.use_lda = True
    testing.var_sensor('LBP',datasettrain='livdet2011',sensortrain ='biometrika',datasettest='livdet2013',sensortest ='italdata')
    #testing.var_sensor('LBP','all','')
    #"""