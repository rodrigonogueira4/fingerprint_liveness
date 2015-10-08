import caffe
import numpy as np

net = caffe.Classifier('/opt/caffe/models/vgg_fingerprint/deploy.prototxt', '/opt/caffe/models/vgg_fingerprint/fingerprint_train_iter_20000.caffemodel')
for key, value in net.params.items():
    #netparams[key] = value[0].data
    np.save(open(key+'.npy','wb'),value[0].data)
    np.save(open(key+'_bias.npy','wb'),value[1].data)