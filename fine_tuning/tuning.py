caffe_root = '/home/jasmine/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()

import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import tempfile

# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

import os
weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
assert os.path.exists(weights)

#Load 1000 imageNet lables and 5 style labels
imagenet_label_file = caffe_root+'data/ilsvrc12/synset_words.txt'
imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))

style_label_file = caffe_root+ 'examples/finetune_flickr_style/style_names.txt'
style_labels = list(np.loadtxt(style_label_file, str, delimiter='\t'))

#define caffenet
from caffe import layers as L
from caffe import params as P

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)]*2

def conv_relu (bottom, ks, nout, stride=1, pad=0, group =1, param = learned_param, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0.1)):
	conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, group=group, param=param, weight_filler=weight_filler, bias_filler=bias_filler)
	return conv, L.ReLU(conv, in_place=True)

def fc_relu (bottom, nout, param = learned_param, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0.1)):
	fc = L.InnerProduct(bottom, num_output = nout, param=param, weight_filler=weight_filler, bias_filler = bias_filler)
	return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
	return L.Pooling(bottom, pool = P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data, label=None, train=True, num_classes=1000, classifier_name='fc8', learn_all=False):
	n = caffe.NetSpec()
	n.data = data
	param = learned_param if learn_all else frozen_param
	n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
	n.pool1 = max_pool(n.relu1, 3, stride=2)
	n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
	n.conv2, n.relu2 = conv_relu(n.norm1,5,256, pad=2, group=2, param=param)
	n.pool2 = max_pool(n.relu2, 3, stride=2)
	n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=.75)
	n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
	n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
	n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
	n.pool5 = max_pool(n.relu5, 3, stride=2)
	n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    
	if train:
		n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
	else:
		fc7input = n.relu6
	n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
	if train:
		n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
	else:
		fc8input = n.relu7
	fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
	

	n.__setattr__(classifier_name, fc8)
	if not train:
		n.probs = L.Softmax(fc8)
	if label is not None:
		n.label = label
		n.loss = L.SoftmaxWithLoss(fc8, n.label)
		n.acc = L.Accuracy(fc8, n.label)
	
	with tempfile.NamedTemporaryFile(delete=False) as f:
		f.write(str(n.to_proto()))
		return f.name

dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227,227]))
imagenet_net_filename = caffenet(data=dummy_data, train=False)
image_net = caffe.Net(imagenet_net_filename, weights, caffe.TEST)

#create stylenet with the same caffenet structure
#diff: 20 classes

def style_net(train=True, learn_all=False, subset=None):
	if subset is None:
		subset = 'train' if train else 'test'
	source = caffe_root + 'data/flickr_style/%s.txt' % subset
	transform_param = dict(mirror=train, crop_size=277, mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
	style_data, style_label = L.ImageData(transform_param=transform_param, source=source, barch_size=50, new_height=256, new_width=256, ntop=2)
	return caffenet(data=style_data, label = style_label, train=train, num_classes=20, classifier_name='fc8_flickr', learn_all=learn_all)

#using caffenet pre-trained params as the init of untrained stylenet
untrained_style_net = caffe.Net(style_net(train=False, subset='train'), weights, caffe.TEST)
untrained_style_net.forward()
style_data_batch = untrained_style_net.blobs['data'].data.copy()
style_label_batch = np.array(untrained_style_net.blobsb['label'].data, dtype=np.int32)


