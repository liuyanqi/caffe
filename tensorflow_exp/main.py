import numpy as np
import tensorflow as tf
import cifar_input
import resnet_model
#import resnet_main

#flags are google way of handling command line arguments
FLAGS = tf.app.flags.FLAGS

#flag_name, default_value, docstring
tf.app.flags.DEFINE_float('lr', 0.1,'learning rate')
tf.app.flags.DEFINE_integer('epoch', 200, 'number of epoches')
tf.app.flags.DEFINE_float('dr', 1, 'data reduction denominator')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval')
tf.app.flags.DEFINE_string('train_dir', '../log/train', 'training directory')
tf.app.flags.DEFINE_string('test_dir', '../log/test', 'testing directory')
tf.app.flags.DEFINE_string('log_root', '../log', 'Directory to keep the checkpoints. should be a parent dirct for train_dir/eval_dir')
tf.app.flags.DEFINE_bool('resume', False, 'resume from checkpoint')

#input
images, labels = cifar_input.build_input('cifar10', './data', FLAGS.epoch, FLAGS.mode)

from keras.datasets import cifar10

(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols = train_features.shape
num_test, _, _, _ = train_features.shape
num_classes = len(np.unique(train_labels))

train_features = np.asfarray(train_features,'float32')
train_features = train_features[0:200, :, :,:]
train_labels_1 = np.zeros((200,10))

train_labels = train_labels[0:200,:]

for i in range(200):
    train_labels_1[i,train_labels[i]] = 1

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Model

if FLAGS.resume:
    #Load checkpoint
    print('==> Resuming from checkpoint')
else:
    print("==> Building model..")

#criterion = tf.nn.softmax_cross_entropy_with_logits()
optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
global_step = tf.contrib.framework.get_or_create_global_step()




def train(hps):
   # the function calls tf.contrib.framework.get_or_create_global_step() and also build_model which builds the resnet graph
    model = resnet_model.ResNet(hps,train_features, train_labels_1, '')
    model._build_model()

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(model.cost, trainable_variables)
 
    
    apply_op = optimizer.apply_gradients(
		zip(grads, trainable_variables),
		global_step=global_step, name='train_step')
    
    train_ops = [apply_op]
    model.train_op = tf.group(*train_ops)
    model.summaries = tf.summary.merge_all()
    
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
		tf.get_default_graph(),
		tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    truth = tf.argmax(model.labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

    summary_hook = tf.train.SummarySaverHook(
			save_steps=100,
			output_dir=FLAGS.train_dir,
			summary_op=tf.summary.merge([model.summaries, tf.summary.scalar('Precision', precision)])
    )

    logging_hook = tf.train.LoggingTensorHook(
		tensors={'step': global_step,
			 'loss': model.cost,
			'precision': precision},
		every_n_iter=100)

    with tf.train.MonitoredTrainingSession(
	checkpoint_dir=FLAGS.log_root,
	hooks=[logging_hook],
	chief_only_hooks=[summary_hook],
	save_summaries_steps=0, #disable the default summary
	config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
	while not mon_sess.should_stop():
		mon_sess.run(model.train_op)  
    

'''
def train(hps):
    model = resnet_model.ResNet(hps, train_features, train_labels, FLAGS.mode)
    model.build_graph()

    truth = tf.argmax(model.lables, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

    summary_hook = tf.train.SummarySaverHook(
			save_steps=100,
			output_dir=FLAGS.train_dir,
			summary_op=tf.summary_merge([model.summaries, tf.summary.scalar('Precision', precision)])
    )

    logging_hook = tf.train.LoggingTensorHook(
		tensors={'step': model.global_step,
			 'loss': model.cost,
			'precision': precision},
		every_n_iter=100)

    with tf.train.MonitoredTrainingSession(
	checkpoint_dir=FLAGS.log_root,
	hooks=[logging_hook],
	chief_only_hook=[summary_hook],
	save_summaries_step=0, #disable the default summary
	config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
	while not mon_sess.should_stop():
		mon_sess.run(model.train_op)


'''
def main(_):
	hps = resnet_model.HParams(batch_size= FLAGS.epoch,
				   num_classes=num_classes,
				   min_lrn_rate=0.0001,
				   lrn_rate=0.1,
				   num_residual_units=5,
				   use_bottleneck=False,
				   weight_decay_rate=0.0002,
				   relu_leakiness=0.1,
				   optimizer='sgd')


	train(hps)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
