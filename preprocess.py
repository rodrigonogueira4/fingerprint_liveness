OS = 'Windows'
import numpy as np
from numpy.random import normal
from datetime import datetime
from scipy import misc
from scipy import ndimage
from convnet import ConvNet
from mrrconvnet import MRRConvNet
from lbp import LBP
from sklearn import preprocessing
from numpy.linalg import svd
from sklearn.preprocessing import scale
from skimage import io
from skimage import color
from multiprocessing import Pool
import sys
import cv2

class PreProcess():

    DEBUG = 0
    feat_extract_name = None
    low_pass =None
    high_pass = None
    gauss_noise = None
    roi = None
    augmentation = None
    size_percentage = None
    feature_extractor = None
    n_processes = 1
    ZCA = False
    
    def __init__(self, feat_extract_name , n_processes, low_pass, high_pass, gauss_noise, roi, size_percentage,\
                 feature_extractor__shape_norm, feature_extractor__shape_conv, \
                 feature_extractor__shape_pool, feature_extractor__n_filters, \
                 feature_extractor__stride_pool, feature_extractor__stoc_pool, \
                 feature_extractor__div_norm, feature_extractor__region_shape, \
                 feature_extractor__region_stride, feature_extractor__top_regions, \
                 feature_extractor__stride_pool_recurrent, feature_extractor__analysis_shape, \
                 feature_extractor__method, \
                 feature_extractor__n_tiles, augmentation, multi_column, aug_rotate \
                 ):
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.gauss_noise = gauss_noise
        self.roi = roi
        self.size_percentage = size_percentage
        self.augmentation = augmentation
        self.aug_rotate = aug_rotate
        self.multi_column = multi_column
        self.feat_extract_name = feat_extract_name
        self.n_processes = n_processes
        
        if feat_extract_name.lower() == 'convnet':
            self.feature_extractor = eval(feat_extract_name+'()')
            self.feature_extractor.n_filters = feature_extractor__n_filters
            self.feature_extractor.shape_norm = feature_extractor__shape_norm
            self.feature_extractor.shape_conv = feature_extractor__shape_conv
            self.feature_extractor.shape_pool = feature_extractor__shape_pool
            self.feature_extractor.stride_pool = feature_extractor__stride_pool
            self.feature_extractor.div_norm = feature_extractor__div_norm
            self.feature_extractor.stoc_pool = feature_extractor__stoc_pool
        elif feat_extract_name.lower() == 'mrrconvnet':
            self.feature_extractor = eval(feat_extract_name+'()')
            convnet = ConvNet()
            convnet.n_filters = feature_extractor__n_filters
            convnet.shape_norm = feature_extractor__shape_norm
            convnet.shape_conv = feature_extractor__shape_conv
            convnet.shape_pool = feature_extractor__shape_pool
            convnet.stride_pool = feature_extractor__stride_pool
            convnet.div_norm = feature_extractor__div_norm
            convnet.stoc_pool = feature_extractor__stoc_pool
            self.feature_extractor.convnet = convnet
            self.feature_extractor.region_shape = feature_extractor__region_shape
            self.feature_extractor.region_stride = feature_extractor__region_stride
            self.feature_extractor.top_regions = feature_extractor__top_regions
            self.feature_extractor.stride_pool_recurrent = feature_extractor__stride_pool_recurrent
            self.feature_extractor.analysis_shape =feature_extractor__analysis_shape
            
        elif feat_extract_name.lower() == 'lbp':
            self.feature_extractor = eval(feat_extract_name+'()')
            self.feature_extractor.method = feature_extractor__method
            self.feature_extractor.n_tiles = feature_extractor__n_tiles  
            
    def get_roi(self, img):
        
        if self.DEBUG >=2:
            antes1 = datetime.now()
        
        imgAux = cv2.morphologyEx(img, cv2.MORPH_OPEN,np.ones((21,21)))
                
        imgAux = imgAux.astype(np.float32)
        
        imgAux = imgAux.max()-imgAux #invert (negate)
        #get geometric center
        yIdx,xIdx = np.indices(imgAux.shape)
        xMean = int(round(np.sum(xIdx * imgAux)/(np.sum(imgAux))))
        yMean = int(round(np.sum(yIdx * imgAux)/(np.sum(imgAux))))
        #get std deviation
        s =1.5
        xStd = int(round(s*((np.sum(((xIdx-xMean)**2)*imgAux)/np.sum(imgAux))**0.5)))
        yStd = int(round(s*((np.sum(((yIdx-yMean)**2)*imgAux)/np.sum(imgAux))**0.5)))

        img = img[max(yMean-yStd,0):min(yMean+yStd,img.shape[0]),max(xMean-xStd,0):min(xMean+xStd,img.shape[1])] #crop
 
        if self.DEBUG>=2:
            print 'tempo ROI =', datetime.now() -antes1
        return img
    
    def get_params(self, deep=True):
        
        params = {}
        params['low_pass'] = self.low_pass
        params['high_pass'] = self.high_pass
        params['gauss_noise'] = self.gauss_noise
        params['roi'] = self.roi
        params['size_percentage'] = self.size_percentage
        params['normalization_per_sample'] = self.normalization_per_sample
        if self.feat_extract_name.lower() == 'convnet':
            params['feature_extractor__n_filters'] = self.feature_extractor.n_filters
            params['feature_extractor__shape_norm'] = self.feature_extractor.shape_norm
            params['feature_extractor__shape_conv'] = self.feature_extractor.shape_conv
            params['feature_extractor__shape_pool'] = self.feature_extractor.shape_pool
            params['feature_extractor__stride_pool'] = self.feature_extractor.stride_pool
            params['feature_extractor__stoc_pool'] = self.feature_extractor.stoc_pool
        elif self.feat_extract_name.lower() == 'mrrconvnet':
            params['feature_extractor__n_filters'] = self.feature_extractor.convnet.n_filters
            params['feature_extractor__shape_norm'] = self.feature_extractor.convnet.shape_norm
            params['feature_extractor__shape_conv'] = self.feature_extractor.convnet.shape_conv
            params['feature_extractor__shape_pool'] = self.feature_extractor.convnet.shape_pool
            params['feature_extractor__stride_pool'] = self.feature_extractor.convnet.stride_pool
            params['feature_extractor__stoc_pool'] = self.feature_extractor.convnet.stoc_pool
            params['feature_extractor__region_shape'] = self.feature_extractor.region_shape
            params['feature_extractor__region_stride'] = self.feature_extractor.region_stride
            params['feature_extractor__stride_pool_recurrent'] = self.feature_extractor.stride_pool_recurrent
            params['feature_extractor__top_regions'] = self.feature_extractor.top_regions
            
        elif self.feat_extract_name.lower() == 'lbp':
            params['feature_extractor__method'] = self.feature_extractor.method
            params['feature_extractor__n_tiles'] = self.feature_extractor.n_tiles
        params['feat_extract_name'] = self.feat_extract_name
        return params
    
    def set_params(self, **params):
        
        for key, value in params.items():
            idx = key.find('__')
            if idx ==-1:
                setattr(self,key,value)
            else:
                #setattr(eval('self.feature_extractor.convnet'),'shape_norm','111')
                setattr(eval('self.'+key[:idx]),key[idx+2:],value)
        return self
    
    def fit(self, X, y=None):#does nothing
        return self
    
    def add_random_noise(self, img):
        img = img.astype(np.float32)
        noise = 255.*normal(0,0.001,img.shape) #mean =0, std=0.001 
        return img+noise
    
    def get_low_pass(self, img):
        img = img.astype(np.float32)
        return ndimage.gaussian_filter(img, sigma=5)
    
    def get_high_pass(self, img):
        img = img.astype(np.float32)
        lowpass = self.get_low_pass(img)
        return img - lowpass
    
    def apply_ZCA(self, X, k=None):
        if k==None:
            k = min(X.shape[0],X.shape[1])
        X = scale(X, axis=1, with_mean=True, with_std=False, copy=False)
        sigma = np.dot(X, X.transpose()) / X.shape[1]
        U,S,V =svd(sigma)
        #xRot = np.dot(U.transpose(), X)
        #xTilde = np.dot(U[:,1:k].transpose(),X)
        epsilon = 0.00001
        xZCAwhite = np.dot(np.dot(U, np.diag(1./(np.diag(S) + epsilon)**0.5)), np.dot(U.transpose(), X))
        return xZCAwhite
    
    final_size = None
    #"""
    def transform(self, X, y=None):
        
        #get the final size to ensure that all images will have this size. This is a workaround for biometrika 2011 sensor since it has images with different sizes
        img = io.imread(X[0])    
        if self.size_percentage<=1.0:
            self.final_size = (np.round(self.size_percentage*np.asarray(img.shape))).astype(np.int)
        else:
            self.final_size = self.size_percentage
        n_processes = min(self.n_processes, len(X))
        p = Pool(processes=n_processes)
        params_pool =[]
        #divide the first estimator parameters among n_jobs
        for i in range(n_processes):
            size = int(np.ceil(float(len(X))/float(n_processes)))
            params_pool.append(X[i*size:min(len(X),(i+1)*size)])
        
        j=len(params_pool)
        results = p.map(unwrap_self_transform_sub, zip([self]*j, params_pool))
        X_out = []
        for result in results:
            X_out.extend(result)
            
        p.close() #terminate process
        p.join() #terminate process
        return np.asarray(X_out)
    #"""
    #X is actually a list of files. Although not elegant, this is efficient because it processes the images on demand
    #def transform(self, X, y=None):
    def transform_sub(self, X, y=None):
        #print 'nao esta rodando em paralelo'
        #img = io.imread(X[0])    
        #self.final_size = (np.round(self.size_percentage*np.asarray(img.shape))).astype(np.int)
        
        X_out = []
        
        #remover depois!!
        rotations = [[0,0,1]]
        if self.augmentation or self.multi_column:
            coordinates = [[[0,0.8],[0,0.8]],[[0.2,1],[0,0.8]],[[0,0.8],[0.2,1]],[[0.2,1],[0.2,1]],[[0.1,0.9],[0.1,0.9]]] #coordinates to do the crop the image
            if self.aug_rotate:
                rotations = [[-5,0.05,0.95],[0,0.05,0.95],[5,0.05,0.95]]
        else:
            coordinates = [[[0.,1],[0.,1.]]] #if not augmented, just crop one image from the center
        
        for filename in X:
        
            if self.DEBUG >=2:
                #print 'file ', i, ' of ', len(X)
                antes1 = datetime.now()
                antes2 = datetime.now()
            
            img = io.imread(filename)
            if self.DEBUG >=2:
                print 'Tempo imread=', datetime.now()-antes2
                
            #convert color images to gray scale
            if len(img.shape) ==3:
                img = (255*color.rgb2gray(io.imread(filename))).astype(np.uint8)
                
            img = img.astype(np.float32)
            
            if self.gauss_noise:
                img = self.add_random_noise(img)
             
            if self.high_pass:
                img = self.get_high_pass(img)
                
            if self.low_pass:    
                img = self.get_low_pass(img) 
                
            if self.roi:
                img = self.get_roi(img) #get only the foreground
                
            for ang,s,e in rotations:    
                
                if self.augmentation:
                    h,w = img.shape 
                    img_rot = misc.imrotate(img, ang)
                    img_rot = img_rot[int(s*h):int(e*h),int(s*w):int(e*w)]
                else:
                    img_rot = img
                    
                for xy1,xy2 in coordinates: #if augmented crop corners and center. If not, just one crop covering the whole image 
                    h,w = img_rot.shape
                    imgs_augmented = []
                    img_crop = img_rot[int(xy1[0]*h):int(xy1[1]*h),int(xy2[0]*w):int(xy2[1]*w)]
                    imgs_augmented.append(img_crop) #append the regular image
                    if self.augmentation or self.multi_column:
                        imgs_augmented.append(img_crop[::,::-1]) #if augmentation, append mirrored image
                    
                    for img_final in imgs_augmented:
                        
                        if self.DEBUG >=2:
                            antes4 = datetime.now()
                        
                        #img_final = img_final[120:-120,100:-100]#Remover depois, fazendo crop forcadamente
                         
                        #get the final size to ensure that all images will have this size. This is a workaround for biometrika 2011 sensor since it has images with different sizes
                        #if final_size ==None:
                        #    final_size = (np.round(self.size_percentage*np.asarray(img_final.shape))).astype(np.int)
                        
                        sample = misc.imresize(img_final, self.final_size)
                        if self.DEBUG >=2:
                            print 'tempo resize=', datetime.now()-antes4
                        
                        if self.feature_extractor != None:
                            if self.DEBUG >=2:
                                antes3 = datetime.now()
                                
                            sample = self.feature_extractor.extract(sample)
                            
                            if self.DEBUG >=2:
                                print 'Tempo feature_extractor = ', datetime.now() - antes3
                                print 'Sample.shape = ', sample.shape
                                sys.stdout.flush()#force prit when running child/sub processes
                        sample = sample.reshape(-1).astype(np.float32)
                        X_out.append(sample)
                
        if self.DEBUG >=1:
            print 'Tempo preprocessing = ', datetime.now() - antes1
            sys.stdout.flush()#force print when running child/sub processes
        X_out = np.asarray(X_out)
        if self.ZCA:
            X_out = self.apply_ZCA(X_out) 
        return X_out

def unwrap_self_transform_sub(arg, **kwarg):
    return PreProcess.transform_sub(*arg, **kwarg)
