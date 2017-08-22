import numpy as np
import matplotlib.pyplot as plt

#set display defaults
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import sys
caffe_root = '../caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe

import os
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print "CaffeNet found"
else:
    print "Caffenet not found"


caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)

mu = np.load('/home/jasmine/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1) #average over pixels to obtain mean pixel values

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data',(2,0,1))
transformer.set_mean('data', mu) #substract the dataset-mean value in each channel
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data', (2,1,0))

#net.blobs['data'].reshape(50, #batch size
#			  3, #3-channel
#			 277, 277) #image size 277*277

image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
#plt.show()

#copy image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

# perform classification
output = net.forward()

output_prob = output['prob'][0]

print 'predicted class is: ', output_prob.argmax()


#load ImageNet labels
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'

labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]

#sort top five predictions
top_inds = output_prob.argsort()[::-1][:5]

print 'probabilities and labels:'
print zip(output_prob[top_inds], labels[top_inds])

