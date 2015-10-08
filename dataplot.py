from matplotlib import pyplot
from testing import Testing
from preprocess import PreProcess
from sklearn.decomposition import RandomizedPCA
if __name__ == '__main__':
    
    """
    testing = Testing()
    testing.divide_by = 5
    testing.n_processes_pproc =3
    lstFilesX,y = testing.load_dataset('Training', 'LivDet2013', 'crossmatch')
    
    #PCA only
    pproc = PreProcess('',1,False,False,False,False,0.25,None,\
        None,None,None,None,None,None,None,None,None)
    X = pproc.transform(lstFilesX)
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(X)
    fig = pyplot.figure()
    pyplot.plot(X[y==False,0],X[y==False,1],'ro')
    pyplot.plot(X[y==True,0],X[y==True,1],'bo')
    pyplot.title('2D Visualization, Crossmatch LivDet 2013 Training, PCA only')
    pyplot.show()
    
    #LBP+PCA
    pproc = PreProcess('LBP',1,False,False,False,False,1.0,None,\
        None,None,None,None,None,None,'uniform',[7,7],False)
    X = pproc.transform(lstFilesX)
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(X)
    fig = pyplot.figure()
    pyplot.plot(X[y==False,0],X[y==False,1],'ro')
    pyplot.plot(X[y==True,0],X[y==True,1],'bo')
    pyplot.title('2D Visualization, Crossmatch LivDet 2013 Training, LBP+PCA')
    pyplot.show()
    
    #ConvNets+PCA
    pproc = PreProcess('ConvNet',1,False,False,False,False,1.0,[(9, 9), (9, 9), (7, 7), (5, 5), (5, 5)],\
        [(9, 9), (9, 9), (7, 7), (5, 5), (5, 5)],[(9, 9), (9, 9), (7, 7), (5, 5), (5, 5)],\
        [64, 128, 256, 512, 1024],[(3,3),(2,2),(2,2),(2,2),(2,2)],False,False,None,None,False)
    X = pproc.transform(lstFilesX)
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(X)
    fig = pyplot.figure()
    pyplot.plot(X[y==False,0],X[y==False,1],'ro')
    pyplot.plot(X[y==True,0],X[y==True,1],'bo')
    pyplot.title('2D Visualization, Crossmatch LivDet 2013 Training, ConvNet 5 Layers+PCA')
    pyplot.show()
    
    
    testing = Testing()
    testing.divide_by = 5
    testing.n_processes_pproc =3
    lstFilesX,y = testing.load_dataset('Testing', 'LivDet2013', 'Crossmatch')
    
    #PCA only
    pproc = PreProcess('',1,False,False,False,False,0.25,None,\
        None,None,None,None,None,None,None,None,None)
    X = pproc.transform(lstFilesX)
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(X)
    fig = pyplot.figure()
    pyplot.plot(X[y==False,0],X[y==False,1],'ro')
    pyplot.plot(X[y==True,0],X[y==True,1],'bo')
    pyplot.title('2D Visualization, Crossmatch LivDet 2013 Testing, PCA only')
    pyplot.show()
    
    
    #LBP+PCA
    pproc = PreProcess('LBP',1,False,False,False,False,1.0,None,\
        None,None,None,None,None,None,'uniform',[7,7],False)
    X = pproc.transform(lstFilesX)
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(X)
    fig = pyplot.figure()
    pyplot.plot(X[y==False,0],X[y==False,1],'ro')
    pyplot.plot(X[y==True,0],X[y==True,1],'bo')
    pyplot.title('2D Visualization, Crossmatch LivDet 2013 Testing, LBP+PCA')
    pyplot.show()
    
    #ConvNets+PCA
    pproc = PreProcess('ConvNet',1,False,False,False,False,1.0,[(9, 9), (9, 9), (7, 7), (5, 5), (5, 5)],\
        [(9, 9), (9, 9), (7, 7), (5, 5), (5, 5)],[(9, 9), (9, 9), (7, 7), (5, 5), (5, 5)],\
        [64, 128, 256, 512, 1024],[(3,3),(2,2),(2,2),(2,2),(2,2)],False,False,None,None,False)
    X = pproc.transform(lstFilesX)
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(X)
    fig = pyplot.figure()
    pyplot.plot(X[y==False,0],X[y==False,1],'ro')
    pyplot.plot(X[y==True,0],X[y==True,1],'bo')
    pyplot.title('2D Visualization, Crossmatch LivDet 2013 Testing, ConvNet 5 Layers+PCA')
    pyplot.show()
    """

    testing = Testing()
    testing.divide_by = 5
    testing.n_processes_pproc =3
    lstFilesX,y = testing.load_dataset('Training', 'LivDet2011', 'digital')
    
    #PCA only
    pproc = PreProcess('',1,False,False,False,False,1.0,None,\
        None,None,None,None,None,None,None,None,None)
    X = pproc.transform(lstFilesX)
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(X)
    fig = pyplot.figure()
    pyplot.plot(X[y==False,0],X[y==False,1],'ro')
    pyplot.plot(X[y==True,0],X[y==True,1],'bo')
    pyplot.title('2D Visualization, Digital LivDet 2011 Training, PCA only')
    pyplot.show()
    
    #LBP+PCA
    pproc = PreProcess('LBP',1,False,False,False,False,1.0,None,\
        None,None,None,None,None,None,'uniform',[7,7],False)
    X = pproc.transform(lstFilesX)
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(X)
    fig = pyplot.figure()
    pyplot.plot(X[y==False,0],X[y==False,1],'ro')
    pyplot.plot(X[y==True,0],X[y==True,1],'bo')
    pyplot.title('2D Visualization, Digital LivDet 2011 Training, LBP+PCA')
    pyplot.show()
    
    #ConvNets+PCA
    pproc = PreProcess('ConvNet',1,False,False,False,False,1.0,[(9,9),(7,7)],[(9,9),(7,7)],[(9,9),(5,5)],[64,256],[(5,5),(4,4)],False,False,None,None,False)
    X = pproc.transform(lstFilesX)
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(X)
    fig = pyplot.figure()
    pyplot.plot(X[y==False,0],X[y==False,1],'ro')
    pyplot.plot(X[y==True,0],X[y==True,1],'bo')
    pyplot.title('2D Visualization, Digital LivDet 2011 Training, ConvNet 2 Layers+PCA')
    pyplot.show()
    
    
    testing = Testing()
    testing.divide_by = 5
    testing.n_processes_pproc =3
    lstFilesX,y = testing.load_dataset('Testing', 'LivDet2011', 'Digital')
    
    #PCA only
    pproc = PreProcess('',1,False,False,False,False,1.0,None,\
        None,None,None,None,None,None,None,None,None)
    X = pproc.transform(lstFilesX)
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(X)
    fig = pyplot.figure()
    pyplot.plot(X[y==False,0],X[y==False,1],'ro')
    pyplot.plot(X[y==True,0],X[y==True,1],'bo')
    pyplot.title('2D Visualization, Digital LivDet 2011 Testing, PCA only')
    pyplot.show()
    
    
    #LBP+PCA
    pproc = PreProcess('LBP',1,False,False,False,False,1.0,None,\
        None,None,None,None,None,None,'uniform',[7,7],False)
    X = pproc.transform(lstFilesX)
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(X)
    fig = pyplot.figure()
    pyplot.plot(X[y==False,0],X[y==False,1],'ro')
    pyplot.plot(X[y==True,0],X[y==True,1],'bo')
    pyplot.title('2D Visualization, Digital LivDet 2011 Testing, LBP+PCA')
    pyplot.show()
    
    #ConvNets+PCA
    pproc = PreProcess('ConvNet',1,False,False,False,False,1.0,[(9,9),(7,7)],[(9,9),(7,7)],[(9,9),(5,5)],[64,256],[(5,5),(4,4)],False,False,None,None,False)
    X = pproc.transform(lstFilesX)
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(X)
    fig = pyplot.figure()
    pyplot.plot(X[y==False,0],X[y==False,1],'ro')
    pyplot.plot(X[y==True,0],X[y==True,1],'bo')
    pyplot.title('2D Visualization, Digital LivDet 2011 Testing, ConvNet 2 Layers+PCA')
    pyplot.show()
