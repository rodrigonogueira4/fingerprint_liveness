import numpy as np
from datetime import datetime
from skimage.util.shape import view_as_windows    
from numpy.random import random_sample
from scipy.ndimage.filters import gaussian_filter
import sys #for stdio  flush
import lcdnorm4
import lpool4

class ConvNet():
    
    print 'Divnorm apenas na primeira camada'
    DEBUG = 0   
    n_filters = None
    shape_norm = None
    shape_conv = None
    shape_pool = None    
    stride_pool = None
    filters_norm = None   
    filters_conv = None
    filters_pooling = None
    #fbstochastic = None
    n_layers = None
    stoc_pool = False
    div_norm = None
    training = True
    retif = True
    
    def extract(self, sample):
        
        #check if the filter bank has the right number of filters
        if self.filters_conv== None:
            self.set_filters()
        for i in range(len(self.n_filters)):
            if self.filters_conv[i].shape[-1] != self.n_filters[i]:
                self.set_filters() #if it does not have the right number of filters, set the new filters
                break 
        
        sample = sample.astype(np.float32)
        sample.shape = sample.shape + (1,)
         
        for n_layer in np.arange(self.n_layers):
                
            if self.DEBUG>=2:
                print 'layer = ', n_layer
                antes1 = datetime.now()
                
            sample = self.convolution(sample,n_layer)
            
            if self.DEBUG>=2:
                print 'convolution=', (datetime.now() - antes1)
                antes1 = datetime.now()
           
            if self.div_norm and n_layer==0:
                #sample = self.subtractive_normalization(sample,self.filters_norm[n_layer])
                #sample = self.divisive_normalization(sample,self.filters_norm[n_layer])
                sample.shape = (1,) + sample.shape
                sample = lcdnorm4.lcdnorm4(sample, self.shape_norm[n_layer], contrast=False, divisive=True)
                sample.shape = sample.shape[1:] 
            
            if self.DEBUG>=2:
                print 'divisive_normalization=', (datetime.now() - antes1)
                antes1 = datetime.now()
           
            if self.stoc_pool:
                sample = self.stochastic_pooling(sample,n_layer)
            else:
                #sample.shape = (1,)+ sample.shape#reshape to four dimensions
                #sample = lpool4.lpool4(sample, neighborhood = self.shape_pool[n_layer], stride=self.stride_pool[n_layer])
                #sample.shape = sample.shape[1:]#reshape to three dimensions
                sample = self.max_pooling(sample,n_layer)
            if self.DEBUG>=2:
                print 'max_pooling=', (datetime.now() - antes1)
        
        if self.DEBUG >=2:       
            print 'final sample.shape=', sample.shape
            sys.stdout.flush()#force print when running child/sub processes
        
        #return sample.reshape(-1)
        return sample
        

    def similarity3D(self, X, fb):
        assert X.ndim == 3
        assert fb.ndim == 4
        assert X.shape[-1] == fb.shape[2]
    
        Xw = view_as_windows(X, fb.shape[:3])
        Xwa = np.abs(Xw-fb)
        return Xwa.sum(axis=(3,4,5))
        
    
    def conv3D(self, X, fb):
        assert X.ndim == 3
        assert fb.ndim == 4
        assert X.shape[-1] == fb.shape[2]
    
        n_filters = fb.shape[-1]
        fb2 = fb.copy()
        fb2.shape = -1, n_filters
    
        X_rfi = view_as_windows(X, fb.shape[:3])
        outh, outw = X_rfi.shape[:2]
        X_rfim = X_rfi.reshape(outh * outw, -1)
        ret = np.dot(X_rfim, fb2) # -- convolution as matrix multiplication
        return ret.reshape(outh, outw, n_filters)
    
    def convolution(self, X, n_layer):
        sample = self.conv3D(X, self.filters_conv[n_layer])
        #sample = self.similarity3D(X, self.filters_conv[n_layer])
        
        # -- post nonlinearities/retified linear
        if self.retif:
            np.maximum(sample,0.,sample) #in place operations are faster because it avoids a copy. That is: a +=b is faster than a = a + b 
            return sample
        else:
            return sample
    
    def max_pooling(self, X, n_layer):
        
        X_rfi = view_as_windows(X, self.filters_pooling[n_layer].shape + (1,))
        X_rfi = X_rfi[::self.stride_pool[n_layer][0], ::self.stride_pool[n_layer][1],:,:,:,:]
        return np.amax(X_rfi, axis=(3,4,5))

    def subtractive_normalization(self, X, filter_norm ):
        # -- pre nonlinearities
        rf_shape_side = (np.asarray(filter_norm.shape) - 1)/2
        if len(X.shape)==2:
            X.shape = X.shape + (1,)
            
        inh, inw, ind = X.shape
        
        ret = self.conv3D(X, filter_norm)
        
        return X[rf_shape_side[0]:inh - rf_shape_side[0], rf_shape_side[1]:inw - rf_shape_side[1], :] - ret


    def divisive_normalization(self, X, filter_norm):
        # -- pre nonlinearities
        rf_shape_side = (np.asarray(filter_norm.shape) - 1)/2
        if len(X.shape)==2:
            X.shape = X.shape + (1,)
            
        inh, inw, _ = X.shape
        
        ret = X ** 2
        
        #ret = self.conv3D(ret, filter_norm) 
        X_rfi = view_as_windows(X, filter_norm.shape[:3])
        ret = X_rfi.sum(axis=(2,3,4))
        # -- post nonlinearities
        ret = np.sqrt(ret)
    
        np.maximum(ret,1.,ret) #avoids that very small numbers will cause the nominator have an greater value
        
        return X[rf_shape_side[0]:inh - rf_shape_side[0], rf_shape_side[1]:inw - rf_shape_side[1], :] / ret

    
    def stochastic_pooling(self, X, n_layer):
        
        inh = X.shape[0] - self.shape_pool[0]+1
        inw = X.shape[1] - self.shape_pool[1]+1
        n_filter = self.n_filters[n_layer]
        filtersize = self.shape_pool[0]*self.shape_pool[1]
    
        randomsamples = random_sample((inh)*(inw)*n_filter).reshape((inh),(inw),n_filter) #generate random values
        randomsamples = np.repeat(randomsamples,repeats = filtersize, axis=2).reshape((inh),(inw),n_filter,filtersize)
     
        X_rfi = view_as_windows(X, self.shape_pool + (1,))
        sumpool = np.repeat(np.sum(X_rfi,axis=(3,4,5)),repeats=filtersize).reshape((inh, inw, n_filter,self.shape_pool[0],self.shape_pool[1],1))
        probabilities = X_rfi/sumpool
        probabilities[np.isnan(probabilities)] = 1/float(filtersize)#get where the sum is zero and replace by one, so the division by zero error do not occur
        probabilities = probabilities.reshape((inh, inw, n_filter,filtersize))
        
        if self.training:
    
            bins = np.add.accumulate(probabilities,axis=3)
            binsbefore = np.concatenate((np.zeros((inh, inw, n_filter,1)),bins[:,:,:,:-1]),axis=3)
            ret = X_rfi[np.where((((binsbefore<= randomsamples) * (bins> randomsamples)))==True)]
            ret = ret.reshape(((inh),(inw),n_filter))[::self.stride_pool,::self.stride_pool]
    
        else: #for testing
            ret = probabilities*X_rfi
            sumpool[sumpool==0.] = 1.
            ret = np.sum(ret,axis=(3))/sumpool[:,:,:,0,0,0]
            ret = ret[::self.stride_pool, ::self.stride_pool]
    
    
    def apply_ZCA(self, Xin):
        X = Xin.copy()
        #ensure data has zero mean
        m = np.mean(X,axis=1).reshape(-1,1)
        X = X-m
        XXt = np.dot(X,np.transpose(X))
        P, D, Pt = np.linalg.svd(XXt)
        D = D**(-1/2)
        D = np.identity(D.shape[0])*D
        W = np.dot(np.dot(P,D),Pt)
        return np.dot(W,X)
    """
    def get_stochastic_filters(self, n_filters):
        filtersize = self.shape_pool[0]*self.shape_pool[1]
        filters =[]
        for n_filter in n_filters:
            #generate random samples for stochastic pooling
            randomsamples = random_sample((inh-12)*(inw-12)*n_filter).reshape((inh-12),(inw-12),n_filter) #generate the random values
            randomsamples = np.repeat(randomsamples,repeats = filtersize, axis=2).reshape((inh-12),(inw-12),n_filter,filtersize)
            
            filters.append(randomsamples)
            
        return filters
    """
    
    def get_random_filtersconv(self, n_filters):
        n_filter_before =1
        filters =[]
        for n_layer in range(len(n_filters)):
            #create filters for convolution
            f_shape = self.shape_conv[n_layer] + (n_filter_before,)
            f_size = f_shape[0] * f_shape[1] * f_shape[2]
            
            # -- set random filter weights
            fbconv = np.random.RandomState(42).randn(f_size, self.n_filters[n_layer]).astype(np.float32)
            # -- subtract mean
            f_mean = fbconv.mean(axis=0)
            fbconv -= f_mean
            # -- set to unit-norm
            f_norm = np.sqrt((fbconv ** 2).sum(axis=0))
            fbconv /= f_norm
            fbconv.shape = f_shape + (self.n_filters[n_layer],)
            filters.append(fbconv)
            n_filter_before = self.n_filters[n_layer]
            
        return filters
    
    def set_filters(self):
    
        #create filters
        self.filters_conv = self.get_random_filtersconv(self.n_filters)
        #self.fbstochastic = self.get_stochastic_filters(self.n_filters)
        
        self.filters_norm = []
        self.filters_pooling = []
        
        aux = np.zeros(self.shape_norm[0])
        aux[(aux.shape[0]-1)/2,(aux.shape[1]-1)/2] = 1 
        gf = gaussian_filter(aux, sigma = 1).astype(np.float32)
        
        #gf.shape = gf.shape+(1,1)
        #self.filters_norm.append(gf)
        for n_layer in range(len(self.n_filters)):
            aux = np.zeros(self.shape_norm[n_layer])
            aux[(aux.shape[0]-1)/2,(aux.shape[1]-1)/2] = 1 
            gf = gaussian_filter(aux, sigma = 1).astype(np.float32)
            gf.shape = gf.shape+(1,)
            gf = np.repeat(gf,self.n_filters[n_layer],axis=2)
            #gf = np.repeat(gf,self.n_filters[n_layer],axis=3)
            self.filters_norm.append(gf)
            self.filters_pooling.append(np.ones(self.shape_pool[n_layer], dtype=np.float32))
        
        self.n_layers = len(self.n_filters)