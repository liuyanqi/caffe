from pylab import *
import matplotlib.pyplot as plt

import sys
caffe_root= '/home/jasmine/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import os

os.chdir(caffe_root)

from caffe import layers as L, params as P

def lenet(lmdb, batch_size):
	n = caffe.NetSpec()
	n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, transform_param=dict(scale=1./255), ntop=2)

	n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
	n.pool1 = L.Pooling(n.conv1, kernel_size=5, stride=2, pool=P.Pooling.MAX)
	n.conv2 = L.Convolution(n.pool1, kernel_size=2, num_output=50, weight_filler=dict(type='xavier'))
	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	n.fc1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
	n.relu1 = L.ReLU(n.fc1, in_place=True)
	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()

#print lenet('mnist_test_lmdb',100)
with open('lenet_auto_train.prototxt', 'w') as f:
	f.write(str(lenet('mnist_train_lmdb',64)))

with open('lenet_auto_test.prototxt', 'w') as f:
	f.write(str(lenet('mnist_test_lmdb',100)))



#caffe.set_device(0)
caffe.set_mode_cpu()

### load the solver and create train and test nets
solver = None
solver = caffe.SGDSolver('/home/jasmine/Desktop/lenet_mnist/lenet_auto_solver.prototxt')

# each output is (batch size, feature dim, spatial dim)
dim = [(k, v.data.shape) for k, v in solver.net.blobs.items()]

solver.net.forward()
solver.test_nets[0].forward()
'''
#train label
plt.figure(1)
imshow(solver.net.blobs['data'].data[:8,0].transpose(1,0,2).reshape(28,8*28), cmap='gray'); axis('off')
print 'train labels:', solver.net.blobs['label'].data[:8]

plt.figure(2)
#test label
imshow(solver.test_nets[0].blobs['data'].data[:8,0].transpose(1,0,2).reshape(28, 8*28), cmap='gray');axis('off')
print 'test labels:', solver.test_nets[0].blobs['label'].data[:8]
plt.show()
'''

#training loop
niter = 200
test_interval = 25
# store loss in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

for it in range(niter):
	solver.step(1)

	#store the train loss
	train_loss[it] = solver.net.blobs['loss'].data
	
	#start the forward pass at conv1 to avoid loading new data
	solver.test_nets[0].forward(start='conv1')
	output[it] = solver.test_nets[0].blobs['score'].data[:8]
	#run a full test every so often
	if it% test_interval ==0:
		print 'Iteration', it, 'testing...'
		correct = 0
		for test_it in range(100):
			solver.test_nets[0].forward()
			correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)==solver.test_nets[0].blobs['label'].data)
		test_acc[int(it/test_interval)] = correct / 1e4

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval*arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:2f}'.format(test_acc[-1]))
plt.show()


