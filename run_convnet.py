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
    testing.n_processes_pproc =3
    testing.use_pca= True
    testing.svm__kernel = 'rbf'
    testing.params_pproc['pproc__size_percentage'] = [0.5]
    testing.params_pproc['pproc__feature_extractor_name'] = ['ConvNet']
    testing.params_auto['pca__n_components'] = [300, 500, 800, 1000]
    testing.params_svm['pred__C'] = [1, 1000, 10000, 100000]
    testing.params_svm['pred__gamma'] = [0.000000001, 0.00000001, 0.0000001, 0.000001]
    testing.params_convnet['pproc__feature_extractor__n_filters'] = [[128]]
    testing.params_convnet['pproc__feature_extractor__stoc_pool'] = [False]
    testing.params_convnet['pproc__feature_extractor__shape_norm'] = [[(9, 9)]]
    testing.params_convnet['pproc__feature_extractor__shape_conv'] = [[(9, 9)]]
    testing.params_convnet['pproc__feature_extractor__shape_pool'] = [[(9, 9)]]
    testing.params_convnet['pproc__feature_extractor__stride_pool'] = [[(9, 9)],[(7, 7)]]
    testing.params_convnet['pproc__feature_extractor__div_norm'] = [False] 
    testing.var_sensor('ConvNet','Griaule13-3','dataset1')
    #testing.params_convnet['pproc__feature_extractor__n_filters'] = [[128,512]]
    #testing.params_convnet['pproc__feature_extractor__stoc_pool'] = [False]
    #testing.params_convnet['pproc__feature_extractor__shape_norm'] = [[(9, 9), (7, 7)]]
    #testing.params_convnet['pproc__feature_extractor__shape_conv'] = [[(9, 9), (7, 7)]]
    #testing.params_convnet['pproc__feature_extractor__shape_pool'] = [[(7, 7), (5, 5)]]
    #testing.params_convnet['pproc__feature_extractor__stride_pool'] = [[(6, 6), (4, 4)]]
    #testing.params_convnet['pproc__feature_extractor__div_norm'] = [False] 
    #testing.var_sensor('ConvNet','Griauli13','dataset1')
    #repeat
    #testing.params_convnet['pproc__feature_extractor__n_filters'] = [[128,512,1024]]
    #testing.params_convnet['pproc__feature_extractor__stoc_pool'] = [False]
    #testing.params_convnet['pproc__feature_extractor__shape_norm'] = [[(9, 9), (7, 7), (5, 5)]]
    #testing.params_convnet['pproc__feature_extractor__shape_conv'] = [[(9, 9), (7, 7), (5, 5)]]
    #testing.params_convnet['pproc__feature_extractor__shape_pool'] = [[(9, 9), (7, 7), (5, 5)]]
    #testing.params_convnet['pproc__feature_extractor__stride_pool'] = [[(4,4), (3,3), (2,2)]]
    #testing.params_convnet['pproc__feature_extractor__div_norm'] = [False]
    #testing.var_sensor('ConvNet','Griauli13','dataset1')
    #repeat
    #testing.params_convnet['pproc__feature_extractor__n_filters'] = [[64,256,512,1024]]
    #testing.params_convnet['pproc__feature_extractor__stoc_pool'] = [False]
    #testing.params_convnet['pproc__feature_extractor__shape_norm'] = [[(9, 9), (9, 9), (7, 7), (5, 5)]]
    #testing.params_convnet['pproc__feature_extractor__shape_conv'] = [[(9, 9), (7, 7), (5, 5), (5, 5)]]
    #testing.params_convnet['pproc__feature_extractor__shape_pool'] = [[(7, 7), (5, 5), (5, 5),(5, 5)]]
    #testing.params_convnet['pproc__feature_extractor__stride_pool'] = [[(3, 3), (2, 2), (2, 2), (2, 2)],[(2, 2), (2, 2), (2, 2), (2, 2)] ]
    #testing.params_convnet['pproc__feature_extractor__div_norm'] = [False]
    #testing.var_sensor('ConvNet','Griauli13','dataset1')
    #repeat
    #testing.params_convnet['pproc__feature_extractor__n_filters'] = [[64,128,256,512,1024]]
    #testing.params_convnet['pproc__feature_extractor__n_filters'] = [[128,256,512,1024,2048]]
    #testing.params_convnet['pproc__feature_extractor__stoc_pool'] = [False]
    #testing.params_convnet['pproc__feature_extractor__shape_norm'] = [[(9, 9), (9, 9), (7, 7), (7, 7), (5, 5)]]
    #testing.params_convnet['pproc__feature_extractor__shape_conv'] = [[(9, 9), (9, 9), (7, 7), (5, 5), (5, 5)]]
    #testing.params_convnet['pproc__feature_extractor__shape_pool'] = [[(9, 9), (9, 9), (7, 7), (5, 5), (5, 5)]]
    #testing.params_convnet['pproc__feature_extractor__stride_pool'] = [[(3, 3), (2, 2), (2, 2), (2, 2), (2, 2)]]
    #testing.params_convnet['pproc__feature_extractor__div_norm'] = [False]
    #testing.var_sensor('ConvNet','LivDet2013','crossmatch')
    
    
    
    testing = Testing()
    testing.size_percentage = (240,320)
    testing.divide_by = 80
    testing.cross_validation = False
    testing.augmentation = False
    testing.aug_rotate = False
    testing.multi_column = False   
    testing.roi = False
    testing.low_pass = False
    testing.high_pass = False
    testing.use_pca = True
    testing.n_processes_pproc =20
    testing.n_processes_cv = 1
    testing.pca__n_components = 800
    testing.svm__gamma = 1e-7
    testing.svm__kernel = 'rbf'
    testing.svm__C = 5000
    testing.stoc_pool = False
    testing.div_norm = False
    #testing.n_filters = [64]
    #testing.shape_norm = [(9, 9)]
    #testing.shape_conv = [(9, 9)]
    #testing.shape_pool = [(9, 9)]
    #testing.stride_pool = [(9, 9)]
    testing.n_filters = [128,512]
    testing.shape_norm = [(9, 9), (7, 7)]
    testing.shape_conv = [(9, 9), (7, 7)]
    testing.shape_pool = [(9, 9), (7, 7)]
    testing.stride_pool = [(6, 6), (4, 4)]
    #testing.n_filters = [128,512,1024]
    #testing.shape_norm = [(9, 9), (7, 7), (5, 5)]
    #testing.shape_conv = [(9, 9), (7, 7), (5, 5)]
    #testing.shape_pool = [(9, 9), (7, 7), (5, 5)]
    #testing.stride_pool = [(4, 4), (3, 3), (2, 2)]
    #testing.n_filters = [64, 256, 512, 1024]
    #testing.shape_norm = [(9, 9), (9, 9), (7, 7), (5, 5)]
    #testing.shape_conv = [(9, 9), (9, 9), (7, 7), (5, 5)]
    #testing.shape_pool = [(7, 7), (5, 5), (5, 5), (5, 5)]
    #testing.stride_pool = [(3, 3), (2, 2), (2, 2), (2, 2)]
    #testing.n_filters = [64,128,256,512,1024]
    #testing.shape_norm = [(9, 9), (9, 9), (7, 7), (5, 5), (5, 5)]
    #testing.shape_conv = [(9, 9), (9, 9), (7, 7), (5, 5), (5, 5)]
    #testing.shape_pool = [(9, 9), (9, 9), (7, 7), (5, 5), (5, 5)]
    #testing.stride_pool = [(3, 3), (2, 2), (2, 2), (2, 2), (2, 2)]
    testing.predict = 'SVM'
    testing.var_sensor('ConvNet',datasettrain='all',sensortrain ='all',datasettest='all',sensortest ='all')
    #"""
    print 'starting'
    testing = Testing()
    testing.convert_dataset_to_txt("LivDet2011", "digital")
    print 'done'