OS = 'Windows'
import numpy as np
from datetime import datetime
from preprocess import PreProcess
from sklearn.metrics import roc_auc_score
from randomizedpca2 import RandomizedPCA2
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FastICA
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from gridsearchcv2 import GridSearchCV2
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
#if OS.lower == 'adesso':
#    from toolbox.fattach import find_attachment_files
import os
import glob
from svc2 import SVC2
from linearsvc2 import LinearSVC2
from sgd2 import SGD2  
import random
from collections import OrderedDict
import pickle
import marshal

class Testing():
    
    DEBUG = 1
    
    if OS.lower() == 'adesso':
        #temp_dir = '/tmp/'
        temp_dir = ''
    elif OS.lower() == 'amazon':
        temp_dir = '/home/ec2-user/tmp/'
    else:
        temp_dir = '/'
        
    datasets = {'LivDet2013': ['crossmatch', 'swipe', 'italdata', 'biometrika'], \
                    'LivDet2011': ['biometrika', 'digital', 'sagem','italdata'], \
                    'LivDet2009': ['biometrika','crossmatch','identix']}
    
    #Possible values for each parameter 
    params_pproc = {} 
    params_pproc['pproc__size_percentage'] = [.25]
    params_convnet = {}
    params_convnet['pproc__feature_extractor__n_filters'] = [[64,128,256],[128,256,512],[256,512,1024]]
    params_convnet['pproc__feature_extractor__stoc_pool'] = [False]
    params_convnet['pproc__feature_extractor__shape_norm'] = [[(9,9),(9,9),(9,9)]]
    params_convnet['pproc__feature_extractor__shape_conv'] = [[(9,9),(9,9),(5,5)]]
    params_convnet['pproc__feature_extractor__shape_pool'] = [[(5,5),(5,5),(5,5)]]
    params_convnet['pproc__feature_extractor__stride_pool'] = [[5,5,2]]
    params_mrrconvnet = {}
    params_mrrconvnet['pproc__feature_extractor.convnet__n_filterss'] = [[64,128,256],[128,256,512],[256,512,1024]]
    params_mrrconvnet['pproc__feature_extractor.convnet__stoc_pool'] = [False]
    params_mrrconvnet['pproc__feature_extractor.convnet__shape_norm'] = [[(9,9),(9,9),(9,9)]]
    params_mrrconvnet['pproc__feature_extractor.convnet__shape_conv'] = [[(9,9),(9,9),(5,5)]]
    params_mrrconvnet['pproc__feature_extractor.convnet__shape_pool'] = [[(5,5),(5,5),(5,5)]]
    params_mrrconvnet['pproc__feature_extractor.convnet__stride_pool'] = [[5,5,2]]
    params_mrrconvnet['pproc__feature_extractor__analysis_shape'] = [(50,50)]
    params_mrrconvnet['pproc__feature_extractor__region_shape'] = [(50,50)]
    params_mrrconvnet['pproc__feature_extractor__stride_pool_recurrent'] = [[(2,2),(2,2)]]
    params_mrrconvnet['pproc__feature_extractor__region_stride'] = [[(2,2),(2,2)]]
    params_mrrconvnet['pproc__feature_extractor__top_regions'] = [5]     
    params_lbp = {}
    params_lbp['pproc__feature_extractor__method'] = ['default', 'uniform']
    params_lbp['pproc__feature_extractor__n_tiles'] = [[1,1],[3,3],[5,5],[7,7]]
    dicfeat_extract = {}
    dicfeat_extract['ConvNet'] = params_convnet
    dicfeat_extract['MRRConvNet'] = params_mrrconvnet
    dicfeat_extract['LBP'] = params_lbp
    
    params_auto = {}
    params_auto['pca__n_components'] = [10, 30, 50, 100, 300]
    #params_auto['lda__n_components'] = [10]
    
    params_svm = {}
    params_svm['pred__C'] = [0.1, 1, 10, 100, 1000, 5000]
    
    params_sgd = {}
    params_sgd['pred__alpha'] = [0.1, 0.001, 0.0001, 0.00001, 0.0000001]
    
    params_knn = {}
    params_knn['pred__weights'] = ['uniform', 'distance']
    params_knn['pred__n_neighbors'] = [1, 3, 9, 15]
    
    dicPredict = {}
    dicPredict['SVM'] = params_svm
    dicPredict['SGD'] = params_sgd
    dicPredict['KNN'] = params_knn
    
    augmentation = None
    aug_rotate = None
    multi_column = None
    cross_validation = None
    divide_by=1
    predict = None
    n_folds = None
    n_processes_cv = None
    n_processes_cv_last_estimator = None
    n_processes_pproc = None
    size_percentage = None
    lbp__n_tiles = None
    lbp__method = None
    n_filters = None
    shape_norm = None
    shape_conv = None
    shape_pool = None    
    stride_pool = None
    div_norm = None
    stoc_pool = None
    analysis_shape = None
    region_shape = None
    region_stride = None
    top_regions = None
    stride_pool_recurrent = None
    svm__gamma = None
    svm__C = None
    svm__kernel = None
    sgd__alpha = 0.0001
    knn__n_neighbors = None
    knn__weights = None
    pca__n_components = None
    use_pca = None
    lda__n_components = None
    use_lda = None
    roi = None
    gauss_noise = None
    high_pass = None
    low_pass = None
    datasettrain = None
    sensortrain = None
    datasettest = None
    sensortest = None
    feat_extract_name = None
    comments = None
    mini_batch_size_test = 1000
    
    def var_sensor(self,feat_extract_name, datasettrain, sensortrain, datasettest=None, sensortest=None):
        print 'Training dataset=',datasettrain,'sensor=',sensortrain
        print 'Testing dataset=',datasettest,'sensor=',sensortest
        self.datasettrain = datasettrain
        if datasettest == None:
            self.datasettest = datasettrain
        else:
            self.datasettest = datasettest
        self.sensortrain = sensortrain
        if sensortest == None:
            self.sensortest = sensortrain
        else:
            self.sensortest = sensortest
        self.feat_extract_name = feat_extract_name
        
        if self.datasettrain.lower()=='all':
            list_files_train,y_train = [],[]
            for datasetname, sensors in self.datasets.items():
                for sensorname in sensors:
                    #skip SWIPE
                    if sensorname.lower() != 'swipe':
                        list_files_trainq,y_trainq = self.load_dataset('Training', datasetname, sensorname)
                        list_files_train.extend(list_files_trainq)
                        y_train.extend(y_trainq)
            y_train = np.asarray(y_train)            
        else:
            list_files_train,y_train = self.load_dataset('Training', self.datasettrain, self.sensortrain)
        
        
        if self.cross_validation:
            list_files_test=None
            y_test=None
        else:
            if self.datasettest.lower()=='all':
                list_files_test,y_test = [],[]
                for datasetname, sensors in self.datasets.items():
                    for sensorname in sensors:
                        #skip SWIPE
                        if sensorname.lower() != 'swipe':
                            list_files_testq,y_testq = self.load_dataset('Testing', datasetname, sensorname)
                            list_files_test.extend(list_files_testq)
                            y_test.extend(y_testq)
                
                y_test = np.asarray(y_test)
            else:
                list_files_test,y_test = self.load_dataset('Testing', self.datasettest, self.sensortest)
                
            
        antes1 = datetime.now()
        score, best_params = self.run_pipeline(feat_extract_name, list_files_train, y_train, list_files_test, y_test)
        
        print 'score=',score
        print 'best_params = '
        if best_params != None:
            for best_param in best_params:
                print best_param
                        
        print 'Time run_pipeline=',datetime.now()-antes1
        print '' 
        return score,best_params
    
    def load_dataset(self, Train_or_Test, dataset, sensor):
                
        if self.DEBUG >=1:
            print 'loading dataset ', Train_or_Test
        
        #get only for training or test and only the specified sensor
        files_aux = []
        
        if OS.lower() == 'adesso':
            #datasetdir = find_attachment_files('p/'+dataset)[0]
            datasetdir = '/awmedia/www/media/p/'+dataset
        elif OS.lower() == 'amazon':
            datasetdir = '/home/ec2-user/' + dataset
        else:
            datasetdir = '/datasets/'+dataset
            
        alldirs = os.walk(datasetdir)
        dirs = []
        #list all files
        for i in alldirs:
            dirs.append(i[0] + '/*.tif')
            dirs.append(i[0] + '/*.bmp')
            dirs.append(i[0] + '/*.png')
        
        for fileglob in dirs:
            files_aux2 = glob.glob(fileglob)
            files_aux2.sort()
            files_aux.extend(files_aux2)    
        files = []
        y = []
        
        random.shuffle(files_aux, random=random.seed(0)) #shuffle the list of files
        for file_aux in files_aux[::self.divide_by]:
            if Train_or_Test.lower() in file_aux.lower():
                if (sensor.lower() in file_aux.lower()) or (sensor.lower() in 'all'):
                    files.append(file_aux.replace('\\', '/'))
                    
                    cat = False #check if it is a false or true finger print
                    if "live" in file_aux.lower():
                        cat = True
                    y.append(cat)
        return files, np.asarray(y)
        #return files[::self.divide_by], np.asarray(y)[::self.divide_by]
    
    def convert_dataset_to_txt(self,  dataset, sensor):
        f = open('train.txt','w')
        files, y = self.load_dataset("Training",dataset, sensor)
        for i in range(len(files)):
            f.write(str(files[i]).replace(" ", "\ ")+" "+str(int(y[i]))+"\n") # python will convert \n to os.linesep
        f.close() # you can omit in most cases as the destructor will call if            
    
        f = open('test.txt','w')
        files, y = self.load_dataset("Testing",dataset,sensor)
        for i in range(len(files)):
            f.write(str(files[i]).replace(" ", "\ ")+" "+str(int(y[i]))+"\n") # python will convert \n to os.linesep
        f.close() # you can omit in most cases as the destructor will call if            
    
    
    def run_pipeline(self, feat_extract_name, X_train, y_train, X_test=None, y_test=None):   

        if self.DEBUG >=1:
            print 'training...'
            antes1 = datetime.now()
    
        pproc= PreProcess(feat_extract_name = feat_extract_name, n_processes = self.n_processes_pproc, size_percentage = self.size_percentage, \
                          roi = self.roi, high_pass = self.high_pass,low_pass = self.low_pass, gauss_noise = self.gauss_noise, \
                          feature_extractor__method = self.lbp__method, feature_extractor__n_tiles = self.lbp__n_tiles, \
                          feature_extractor__n_filters = self.n_filters, feature_extractor__shape_norm = self.shape_norm, \
                          feature_extractor__shape_conv = self.shape_conv, feature_extractor__shape_pool = self.shape_pool, \
                          feature_extractor__stride_pool = self.stride_pool, feature_extractor__stoc_pool = self.stoc_pool, \
                          feature_extractor__div_norm = self.div_norm, \
                          feature_extractor__region_shape = self.region_shape, feature_extractor__region_stride = self.region_stride, \
                          feature_extractor__top_regions = self.top_regions, feature_extractor__stride_pool_recurrent = self.stride_pool_recurrent, \
                          feature_extractor__analysis_shape = self.analysis_shape, multi_column = self.multi_column, \
                          augmentation = self.augmentation, aug_rotate = self.aug_rotate
                          )
        norm = preprocessing.StandardScaler(copy=True)
        
        piplist = []
        if self.cross_validation:
            piplist.append(('pproc', pproc))
        piplist.append(('norm', norm))
        
        if self.use_pca:
            pca = RandomizedPCA2(whiten=True,random_state=0, n_components=self.pca__n_components, copy=True) #Must use fit_transform instead of fit() and then transform() when copy=false
            #from sklearn.decomposition import PCA
            #pca = PCA(whiten=True, n_components=self.pca__n_components, copy=True)
            #pca = FastICA(whiten=True,random_state=0, n_components=self.pca__n_components, max_iter=400)
            #pca = SparsePCA(random_state=0, n_components=self.pca__n_components) #Must use fit_transform instead of fit() and then transform() when copy=false
            piplist.append(('pca', pca))
        
        if self.use_lda:
            lda = LDA(n_components=self.lda__n_components)
            piplist.append(('lda', lda))
            
        if self.predict.lower() =='svm':
            if self.svm__kernel.lower() =='rbf':
                pred = SVC2(kernel='rbf', class_weight='auto', random_state=0, C=self.svm__C, gamma=self.svm__gamma, multi_column = self.multi_column, augmentation = self.augmentation, aug_rotate = self.aug_rotate)
            else:
                pred = LinearSVC2(random_state=0, fit_intercept=False, class_weight='auto', C=self.svm__C, augmentation = self.augmentation)
        elif self.predict.lower() =='sgd':
            pred = SGD2(loss="hinge", penalty="l2", l1_ratio=0.05, random_state=0, n_iter=5, shuffle=True, augmentation = self.augmentation, alpha = self.sgd__alpha)
        elif self.predict.lower() =='knn':
            pred = KNeighborsClassifier(n_neighbors=self.knn__n_neighbors, weights=self.knn__weights)
        piplist.append(('pred', pred))
        
        pipeline = Pipeline(piplist)
        
        if self.cross_validation:
            params_grid = self.params_auto.copy()
            params_grid.update(self.dicPredict[self.predict])
            params_grid.update(self.params_pproc)
            if feat_extract_name.lower() != 'none':
                params_grid.update(self.dicfeat_extract[feat_extract_name])
            
            pipelineGridSearch = GridSearchCV2(pipeline, params_grid,cv=self.n_folds, verbose=0, n_jobs=self.n_processes_cv, n_jobs_last_estimator = self.n_processes_cv_last_estimator, augmentation = self.augmentation, auto_adjust_params = None, testing = self)
            pipelineGridSearch.fit(X_train,y_train)
            
            #gridsearchRef = GridSearchCV(pipeline, params_grid, cv=self.n_folds, iid=True, scoring = 'roc_auc', verbose=0, n_jobs=1)
            #gridsearchRef.fit(X_train,y_train)
            #print 'ReF=== score=', gridsearchRef.best_score_,'params=', gridsearchRef.best_params_
            
            return pipelineGridSearch.best_score_, pipelineGridSearch.best_params_
        else:
            antes= datetime.now()
            
            X_train = pproc.transform(X_train)
            antes2= datetime.now()
            X_test = pproc.transform(X_test)
            time_pproc=datetime.now()-antes2
            if self.multi_column:
                
                y_pred_train = []
                y_pred_test = []
                if self.aug_rotate:
                    multiply = 30
                else:
                    multiply = 10
                    
                for i in range(multiply):
                    pipeline = pipeline.fit(X_train[i::multiply,:],y_train)
                    y_pred_train.append(pipeline.predict(X_train[i::multiply,:]))
                    y_pred_test.append(pipeline.predict(X_test[i::multiply,:]))
                
                y_pred_train = np.mean(np.asarray(y_pred_train),axis=0)
                y_pred_train[y_pred_train>=0]=1
                y_pred_train[y_pred_train<0]=0
                
                y_pred_test = np.mean(np.asarray(y_pred_test),axis=0)
                y_pred_test[y_pred_test>=0]=1
                y_pred_test[y_pred_test<0]=0   
            else:
                pipeline = pipeline.fit(X_train,y_train)
                
                #save the classifier
                with open(self.temp_dir+'clf_'+self.datasettrain.lower() + '_' + self.sensortrain.lower() + '_' + self.feat_extract_name.lower() + '.pkl', 'wb') as output:
                    pickle.dump(pipeline, output, pickle.HIGHEST_PROTOCOL)
                    
                y_pred_train = pipeline.predict(X_train)
                
                antes2= datetime.now()    
                y_pred_test = []
                for i in range(0,len(X_test),self.mini_batch_size_test):  
                    y_pred_test.extend(list(pipeline.predict(X_test[i:i+self.mini_batch_size_test])))
                
                test_time = (datetime.now() - antes2)+time_pproc
                
                print 'Tempo Predict= ', test_time # DEBUG
                print 'Numero de amostras', str(len(X_test))
                    
            score_training = 100.-(100.*roc_auc_score(y_train, y_pred_train))
            print 'score_training=', score_training

            score = 100.-(100.*roc_auc_score(y_test, np.asarray(y_pred_test)))
            total_time = datetime.now() - antes  # DEBUG
            
            pca = pipeline.steps[-2][1]
            pca_total_variance= None
            if hasattr(pca, 'explained_variance_ratio_'):
                pca_total_variance= np.sum(pca.explained_variance_ratio_)
            pred = pipeline.steps[-1][1]
            n_support_=None
            if hasattr(pred, 'n_support_'):
                n_support_ = pred.n_support_
            
            self.append_results(params=None, score_mean=score, score_std=None, total_time=total_time, test_time=test_time,n_test_samples=str(len(X_test)), score_training=score_training, n_svm_vectors=n_support_, pca_total_variance=pca_total_variance)
            return score, None
        
        if self.DEBUG >=1:
            print 'Tempo Fit Pipeline= ', (datetime.now() - antes1)  # DEBUG

    
    def append_results(self, params, score_mean, score_std, total_time, test_time=None, n_test_samples=None, score_training=None, n_svm_vectors=None, pca_total_variance=None):
        dic_params = OrderedDict()
        dic_params['pproc__feature_extractor_name'] = self.feat_extract_name
        dic_params['classifier_name'] = self.predict
        dic_params['use_pca'] = self.use_pca
        dic_params['datasettrain'] = self.datasettrain
        dic_params['sensortrain'] = self.sensortrain
        dic_params['datasettest'] = self.datasettest
        dic_params['sensortest'] = self.sensortest
        dic_params['cross_validation'] = self.cross_validation
        dic_params['augmentation'] = self.augmentation
        dic_params['aug_rotate'] = self.aug_rotate
        dic_params['multi_column'] = self.multi_column
        dic_params['pproc__size_percentage'] = self.size_percentage
        dic_params['pproc__roi'] = self.roi
        dic_params['pproc__low_pass'] = self.low_pass
        dic_params['pproc__high_pass'] = self.high_pass
        dic_params['pproc__gauss_noise'] = self.gauss_noise
        dic_params['pproc__feature_extractor__method'] = self.lbp__method
        dic_params['pproc__feature_extractor__n_tiles'] = self.lbp__n_tiles
        dic_params['pproc__feature_extractor__n_filters'] = self.n_filters
        dic_params['pproc__feature_extractor__shape_norm'] = self.shape_norm
        dic_params['pproc__feature_extractor__shape_conv'] = self.shape_conv
        dic_params['pproc__feature_extractor__shape_pool'] = self.shape_pool
        dic_params['pproc__feature_extractor__stride_pool'] = self.stride_pool
        dic_params['pproc__feature_extractor__stoc_pool'] = self.stoc_pool
        dic_params['pproc__feature_extractor__div_norm'] = self.div_norm
        dic_params['pca__n_components'] = self.pca__n_components
        dic_params['lda__n_components'] = self.lda__n_components
        dic_params['pred__C'] = self.svm__C
        dic_params['pred__kernel'] = self.svm__kernel
        dic_params['pred__gamma'] = self.svm__gamma
        dic_params['pred__weights'] = self.knn__weights
        dic_params['pred__n_neighbors'] = self.knn__n_neighbors
        dic_params['use_lda'] = self.use_lda
        dic_params['n_folds'] = self.n_folds
        dic_params['divide_by'] = self.divide_by
        dic_params['datetime'] = str(datetime.now().strftime('%Y%m%d-%H%M%S'))
        dic_params['total_time'] = total_time
        dic_params['test_time'] = test_time
        dic_params['n_test_samples'] = n_test_samples
        dic_params['score_mean'] = score_mean
        dic_params['score_std'] = score_std
        dic_params['score_training'] = score_training
        dic_params['n_svm_vectors'] = n_svm_vectors
        dic_params['pca_total_variance'] = pca_total_variance
        dic_params['comments'] = self.comments
        output_path = self.temp_dir  +'results.csv'
        
        f = open(output_path, 'a') #a for appending
        
        #write header if file is empty
        if self.is_zero_file(output_path):
            header = ''
            for key in dic_params.iterkeys():
                header += key+';'
            header+='\n'
            f.write(header)
        
        #get the modified parameters only and put in the dic_params
        if params!=None:
            for key,value in params.items():
                dic_params[key] = value
            
        line = ''
        for value in dic_params.itervalues():
            line += str(value)+';'
        
        line +='\n'
        f.write(line)
        f.close()
        
    def is_zero_file(self, fpath):  
        return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True
