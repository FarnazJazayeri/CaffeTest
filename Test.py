import numpy as np
import caffe


def sayHello(text):
    print('hello'+text+'!')
    
    return text

sayHello('Farnaz')

caffe.set_mode_cpu()

net=caffe.Net('conv.prototxt',caffe.TEST)
print net.blobs['conv'].data.shape
