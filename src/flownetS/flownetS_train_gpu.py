from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import flownet_S as flowNet
import flownet_input 
from six.moves import xrange  # pylint: disable=redefined-builtin
import os.path
import re
import time
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('log_dir', '/mnt/flownet/log/flownetSl/train', """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('log_dir', './train_weight/flownetS', """Directory where to write event logs and checkpoint.""")
#tf.app.flags.DEFINE_string('checkpoint_dir', '/mnt/flownet/log/flownetSl/train', """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './train_weight/flownetS', """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('data_dir', './data', """Directory where to get data list.""")
#tf.app.flags.DEFINE_integer('max_steps', 1200000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_steps', 3000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 8, """Batch size.""")
tf.app.flags.DEFINE_integer('num_gpus', 1, """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
def tower_loss(scope):
	"""Calculate the total loss on a single tower running the CIFAR model.

	Args:
	scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

	Returns:
	 Tensor of shape [] containing the total loss for a batch of data
	"""
	# Get images and flows for Flownet.
  	img1, img2, flo = flownet_input.inputs(False, FLAGS.data_dir, FLAGS.batch_size)

	# Build a Graph that computes predictions from the inference model.
	logits = flowNet.inference(img1, img2, FLAGS.batch_size)

	# Add to the Graph the Ops for loss calculation.
	_ = flowNet.loss(logits, flo)

	# Assemble all of the losses for the current tower only.
	losses = tf.get_collection('losses', scope)

	# Calculate the total loss for the current tower.
	total_loss = tf.add_n(losses, name='total_loss')

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
		# session. This helps the clarity of presentation on tensorboard.
		loss_name = re.sub('%s_[0-9]*/' % flowNet.TOWER_NAME, '', l.op.name)
		tf.summary.scalar(loss_name, l)

	return total_loss


def average_gradients(tower_grads):
	"""Calculate the average gradient for each shared variable across all towers.

	Note that this function provides a synchronization point across all towers.

	Args:
	tower_grads: List of lists of (gradient, variable) tuples. The outer list
	  is over individual gradients. The inner list is over the gradient
	  calculation for each tower.
	Returns:
	 List of pairs of (gradient, variable) where the gradient has been averaged
	 across all towers.
	"""
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)

			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)
		
		# Average over the 'tower' dimension.
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads

def train():
	"""Train Flownet for a number of steps."""

	# Tell TensorFlow that the model will be built into the default Graph.
	with tf.Graph().as_default(), tf.device('/cpu:0'):


		global_step = tf.Variable(0, trainable=False)

		#boundaries = [300000, 400000, 500000]
		#values = [0.0001, 0.00005, 0.000025, 0.0000125]#S
		#boundaries = [5000*2, 10000*2, 400000*2, 600000*2, 800000*2, 1000000*2]
		#values = [0.000001,0.00001, 0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625]#C
		boundaries = [400000, 600000, 800000, 1000000]
		values = [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625]#Sl

		learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

		# Create an optimizer that performs gradient descent.
		opt = tf.train.AdamOptimizer(learning_rate)

		# Calculate the gradients for each model tower.
		tower_grads = []
		with tf.variable_scope(tf.get_variable_scope()):
			for i in xrange(FLAGS.num_gpus):
				with tf.device('/gpu:%d' % i):
					with tf.name_scope('%s_%d' % (flowNet.TOWER_NAME, i)) as scope:
						# Calculate the loss for one tower of the model. This function
						# constructs the entire CIFAR model but shares the variables across
						# all towers.
						loss = tower_loss(scope)

						# Reuse variables for the next tower.
						tf.get_variable_scope().reuse_variables()

						# Retain the summaries from the final tower.
						summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

						# Calculate the gradients for the batch of data on this tower.
						grads = opt.compute_gradients(loss,var_list=tf.trainable_variables())
						# Keep track of the gradients across all towers.
						tower_grads.append(grads)

		# We must calculate the mean of each gradient. Note that this is the
		# synchronization point across all towers.
		grads = average_gradients(tower_grads)

		# Add a summary to track the learning rate.
		summaries.append(tf.summary.scalar('learning_rate', learning_rate))

		# Add histograms for gradients.
		for grad, var in grads:
			if grad is not None:
				summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

		# Apply the gradients to adjust the shared variables.
		train_op = opt.apply_gradients(grads, global_step=global_step)
		
		#Add histograms for trainable variables.
		for var in tf.trainable_variables():
			summaries.append(tf.summary.histogram(var.op.name, var))
		
		# Build the summary Tensor based on the TF collection of Summaries.
		summary_op = tf.summary.merge(summaries)

		# Add the variable initializer Op.
		init = tf.global_variables_initializer()
		
		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver(tf.global_variables())

		# Create a session for running Ops on the Graph.
		sess = tf.Session()

		# Instantiate a SummaryWriter to output summaries and the Graph.
		summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

		# And then after everything is built:
		config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=FLAGS.log_device_placement)
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.9
		sess = tf.Session(config=config)
		# Run the Op to initialize the variables.
		sess.run(init)
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver_restore = tf.train.Saver(tf.get_collection('fix'))
			saver_restore.restore(sess, ckpt.model_checkpoint_path)
		else:
			print('No checkpoint file found')
			
		# Start the queue runners.
		tf.train.start_queue_runners(sess=sess)
		
		# Start the training loop.
		for step in xrange(FLAGS.max_steps):
			start_time = time.time()
			_, loss_value = sess.run([train_op, loss])
			duration = time.time() - start_time

			assert not np.isnan(loss_value) , 'Model diverged with loss = NaN'

			#Print an overview fairly often.
			if step % 10 == 0:
				num_examples_per_step = FLAGS.batch_size
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = duration

				format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ''sec/batch)')
				print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

			# Write the summaries. 
			if step % 1000 == 0:
				# Print status to stdout.
				print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
				# Update the events file.
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

			# Save a checkpoint.
			if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
				saver.save(sess, checkpoint_file, global_step=step)

def main(argv=None):  # pylint: disable=unused-argument
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	train()


if __name__ == '__main__':
	tf.app.run()
