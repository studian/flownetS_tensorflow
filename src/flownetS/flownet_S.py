from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import re

TOWER_NAME = 'tower'

def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.
	Args:
	name: name of the variable
	shape: list of ints
	initializer: initializer for Variable
	Returns:
	Variable Tensor
	"""
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer)
	return var

def _variable_with_weight_decay(name, shape, stddev, wd):
	"""Helper to create an initialized Variable with weight decay.
	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.
	Args:
	name: name of the variable
	shape: list of ints
	stddev: standard deviation of a truncated Gaussian
	wd: add L2Loss weight decay multiplied by this float. If None, weight
		decay is not added for this Variable.
	Returns:
	Variable Tensor
	"""
	var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def _activation_summary(x):
	"""Helper to create summaries for activations.

	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.

	Args:
	x: Tensor
	Returns:
	nothing
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def inference(images1, images2, batch_size):
	"""Build the FLOWNETS model up to where it may be used for inference.

	Args:
		images: Images placeholder, from inputs().
	Returns:
		predict flows
	"""
	
	images= tf.concat([images1, images2], 3)

	# conv1
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[7, 7, 6, 64], stddev=math.sqrt(2.0 / (7 * 7 * 6)), wd = 0.0004)
		conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv1)
		#print(conv1)
	# conv2
	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 128], stddev=math.sqrt(2.0 / (5 * 5 * 64)), wd = 0.0004)
		conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv2)
		#print(conv2)
	# conv3
	with tf.variable_scope('conv3') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[5, 5, 128, 256], stddev=math.sqrt(2.0 / (5 * 5 * 128)), wd = 0.0004)
		conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv3)
		#print(conv3)
	# conv3_1
	with tf.variable_scope('conv3_1') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256], stddev=math.sqrt(2.0 / (3 * 3 * 256)), wd = 0.0004)
		conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv3_1 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv3_1)
		
	# conv4
	with tf.variable_scope('conv4') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 512], stddev=math.sqrt(2.0 / (3 * 3 * 256)), wd = 0.0004)
		conv = tf.nn.conv2d(conv3_1, kernel, [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv4)
	
	# conv4_1
	with tf.variable_scope('conv4_1') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=math.sqrt(2.0 / (3 * 3 * 512)), wd = 0.0004)
		conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv4_1 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv4_1)

	# conv5
	with tf.variable_scope('conv5') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=math.sqrt(2.0 / (3 * 3 * 512)), wd = 0.0004)
		conv = tf.nn.conv2d(conv4_1, kernel, [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv5)
	
	# conv5_1
	with tf.variable_scope('conv5_1') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=math.sqrt(2.0 / (3 * 3 * 512)), wd = 0.0004)
		conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv5_1 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv5_1)

	# conv6
	with tf.variable_scope('conv6') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 1024], stddev=math.sqrt(2.0 / (3 * 3 * 512)), wd = 0.0004)
		conv = tf.nn.conv2d(conv5_1, kernel, [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv6 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv6)

	# conv6_1
	with tf.variable_scope('conv6_1') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[3, 3, 1024, 1024], stddev=math.sqrt(2.0 / (3 * 3 * 1024)), wd = 0.0004)
		conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv6_1 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv6_1)

	# predict_flow6
	with tf.variable_scope('predict_flow6') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[3, 3, 1024, 2], stddev=math.sqrt(2.0 / (3 * 3 * 1024)), wd = 0.0004)
		conv = tf.nn.conv2d(conv6_1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
		predict_flow6 = tf.nn.bias_add(conv, biases, name=scope.name)
		_activation_summary(predict_flow6)

	# deconv5
	with tf.variable_scope('deconv5') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[4, 4, 512, 1024], stddev=math.sqrt(2.0 / (4 * 4 * 1024)), wd = 0.0)
		shape = conv5_1.get_shape().as_list()
		height = shape[1]
		width = shape[2]
		deconv = tf.nn.conv2d_transpose(conv6_1, kernel, [batch_size, height, width, 512], [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(deconv, biases)
		deconv5 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(deconv5)

	#upsample_flow6to5
	with tf.variable_scope('upsample_flow6to5') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[4, 4, 2, 2], stddev=math.sqrt(2.0 / (4 * 4 * 2)), wd = 0.0)
		shape = conv5_1.get_shape().as_list()
		height = shape[1]
		width = shape[2]
		deconv = tf.nn.conv2d_transpose(predict_flow6, kernel, [batch_size, height, width, 2], [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
		upsample_flow6to5 = tf.nn.bias_add(deconv, biases, name=scope.name)
		_activation_summary(upsample_flow6to5)

	concat5 = tf.concat([upsample_flow6to5, deconv5, conv5_1], 3)

	# predict_flow5
	with tf.variable_scope('predict_flow5') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[3, 3, 2+512+512, 2], stddev=math.sqrt(2.0 / (3 * 3 * 1026)), wd = 0.0004)
		conv = tf.nn.conv2d(concat5, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
		predict_flow5 = tf.nn.bias_add(conv, biases, name=scope.name)
		_activation_summary(predict_flow5)

	# deconv4
	with tf.variable_scope('deconv4') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[4, 4, 256, 1026], stddev=math.sqrt(2.0 / (4 * 4 * 1026)), wd = 0.0)
		shape = conv4_1.get_shape().as_list()
		height = shape[1]
		width = shape[2]
		deconv = tf.nn.conv2d_transpose(concat5, kernel, [batch_size, height, width, 256], [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(deconv, biases)
		deconv4 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(deconv4)

	#upsample_flow5to4
	with tf.variable_scope('upsample_flow5to4') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[4, 4, 2, 2], stddev=math.sqrt(2.0 / (4 * 4 * 2)), wd = 0.0)
		shape = conv4_1.get_shape().as_list()
		height = shape[1]
		width = shape[2]
		deconv = tf.nn.conv2d_transpose(predict_flow5, kernel, [batch_size, height, width, 2], [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
		upsample_flow5to4 = tf.nn.bias_add(deconv, biases, name=scope.name)
		_activation_summary(upsample_flow5to4)

	concat4 = tf.concat([upsample_flow5to4, deconv4, conv4_1], 3)

	# predict_flow4
	with tf.variable_scope('predict_flow4') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[3, 3, 2+256+512, 2], stddev=math.sqrt(2.0 / (3 * 3 * 770)), wd = 0.0004)
		conv = tf.nn.conv2d(concat4, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
		predict_flow4 = tf.nn.bias_add(conv, biases, name=scope.name)
		_activation_summary(predict_flow4)

	# deconv3
	with tf.variable_scope('deconv3') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[4, 4, 128, 770], stddev=math.sqrt(2.0 / (4 * 4 * 770)), wd = 0.0)
		shape = conv3_1.get_shape().as_list()
		height = shape[1]
		width = shape[2]
		deconv = tf.nn.conv2d_transpose(concat4, kernel, [batch_size, height, width, 128], [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(deconv, biases)
		deconv3 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(deconv3)

	#upsample_flow4to3
	with tf.variable_scope('upsample_flow4to3') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[4, 4, 2, 2], stddev=math.sqrt(2.0 / (4 * 4 * 2)), wd = 0.0)
		shape = conv3_1.get_shape().as_list()
		height = shape[1]
		width = shape[2]
		deconv = tf.nn.conv2d_transpose(predict_flow4, kernel, [batch_size, height, width, 2], [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
		upsample_flow4to3 = tf.nn.bias_add(deconv, biases, name=scope.name)
		_activation_summary(upsample_flow4to3)

	concat3 = tf.concat([upsample_flow4to3, deconv3, conv3_1], 3)

	# predict_flow3
	with tf.variable_scope('predict_flow3') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[3, 3, 2+128+256, 2], stddev=math.sqrt(2.0 / (3 * 3 * 386)), wd = 0.0004)
		conv = tf.nn.conv2d(concat3, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
		predict_flow3 = tf.nn.bias_add(conv, biases, name=scope.name)
		_activation_summary(predict_flow3)

	# deconv2
	with tf.variable_scope('deconv2') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[4, 4, 64, 386], stddev=math.sqrt(2.0 / (4 * 4 * 386)), wd = 0.0)
		shape = conv2.get_shape().as_list()
		height = shape[1]
		width = shape[2]
		deconv = tf.nn.conv2d_transpose(concat3, kernel, [batch_size, height, width, 64], [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(deconv, biases)
		deconv2 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(deconv2)

	#upsample_flow3to2
	with tf.variable_scope('upsample_flow3to2') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[4, 4, 2, 2], stddev=math.sqrt(2.0 / (4 * 4 * 2)), wd = 0.0)
		shape = conv2.get_shape().as_list()
		height = shape[1]
		width = shape[2]
		deconv = tf.nn.conv2d_transpose(predict_flow3, kernel, [batch_size, height, width, 2], [1, 2, 2, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
		upsample_flow3to2 = tf.nn.bias_add(deconv, biases, name=scope.name)
		_activation_summary(upsample_flow3to2)

	concat2 = tf.concat([upsample_flow3to2, deconv2, conv2], 3)

	# predict_flow2
	with tf.variable_scope('predict_flow2') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[3, 3, 2+64+128, 2], stddev=math.sqrt(2.0 / (3 * 3 * 194)), wd = 0.0004)
		conv = tf.nn.conv2d(concat2, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
		predict_flow2 = tf.nn.bias_add(conv, biases, name=scope.name)
		_activation_summary(predict_flow2)
		#print(predict_flow2)
	return [predict_flow6, predict_flow5, predict_flow4, predict_flow3, predict_flow2]
	#return predict_flow6

def compute_euclidean_distance(x, y):
	"""
	Computes the euclidean distance between two tensorflow variables
	"""
	d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), 3))
	meand = tf.reduce_mean(d, [1,2])
	return meand

def get_size(x):
	shape = x.get_shape().as_list()
	return [shape[1], shape[2]]

def loss(logits, flos):
	"""Add endpoint error (EPE) predict flows to ground truth with different weight  
	Args:
    logits: Predict flows from inference().
    flos: Grund truth

  	Returns:
    	Loss tensor of type float.
	"""
	x = logits[0]
	y = tf.image.resize_images(flos, get_size(x))
	flow6_loss = tf.scalar_mul(0.32, tf.reduce_mean(compute_euclidean_distance(x,y)))

	x = logits[1]
	y = tf.image.resize_images(flos, get_size(x))
	flow5_loss = tf.scalar_mul(0.08, tf.reduce_mean(compute_euclidean_distance(x,y)))

	x = logits[2]
	y = tf.image.resize_images(flos, get_size(x))
	flow4_loss = tf.scalar_mul(0.02, tf.reduce_mean(compute_euclidean_distance(x,y)))

	x = logits[3]
	y = tf.image.resize_images(flos, get_size(x))
	flow3_loss = tf.scalar_mul(0.01, tf.reduce_mean(compute_euclidean_distance(x,y)))

	x = logits[4]
	y = tf.image.resize_images(flos, get_size(x))
	flow2_loss = tf.scalar_mul(0.005, tf.reduce_mean(compute_euclidean_distance(x,y)))

	tf.add_to_collection('losses', tf.add_n([flow6_loss, flow5_loss, flow4_loss, flow3_loss, flow2_loss]))
	
	"""
	x = logits
	y = tf.image.resize_images(flos, get_size(x))
	flow6_loss = tf.scalar_mul(0.32, tf.reduce_mean(compute_euclidean_distance(x,y)))
	tf.add_to_collection('losses', flow6_loss)
	"""
	return tf.add_n(tf.get_collection('losses'), name='total_loss')






