from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import flownet_S as flowNet
import flownet_input 
import time
import math
from datetime import datetime
from lib import flowlib
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', '/mnt/flownet/log/flownetSl/test', """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', './data', """Directory where to get data list.""")
tf.app.flags.DEFINE_integer('batch_size', 8, """Batch size.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/mnt/flownet/log/flownetSl/train', """Directory where to read model checkpoints.""")
#tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, """How often to run the eval.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 30, """How often to run the eval.""")
def eval_once(saver, summary_writer, loss, x, y, summary_op):
	"""Run Eval once.

	Args:
	saver: Saver.
	summary_writer: Summary writer.
	loss: Calculate EPE
	summary_op: Summary op.
	"""
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	#config.gpu_options.per_process_gpu_memory_fraction = 0.5
	with tf.Session(config=config) as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks something like:
			#   /my-favorite-path/cifar10_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
			print('checkpoint file found')
		else:
			print('No checkpoint file found')
			return

		# Start the queue runners.
		start_time = time.time()
		
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

			num_iter = int(math.ceil(flownet_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size))
			EPE = 0.0
			step = 0
			while step < num_iter and not coord.should_stop():
				flo, pflo, predictions = sess.run([x, y, loss])
				EPE += predictions
				step += 1
			flowlib.write_flow(pflo[-1], "p.flo")
			flowlib.write_flow(flo[-1], "gt.flo")
			precision = EPE / num_iter #/ flownet_input.IMAGE_HEIGHT / flownet_input.IMAGE_WIDTH
			print('%s: EPE = %.3f' % (datetime.now(), precision))

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='EPE', simple_value=precision)
			summary_writer.add_summary(summary, global_step)

		except Exception as e:  # pylint: disable=broad-except
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)
		
		duration = time.time()-start_time
		print("fps = {0:.3},  {1:.3}(sec/frame)".format(1*flownet_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/duration, duration/flownet_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL))

def test():
	"""Train Flownet for a number of steps."""

	# Tell TensorFlow that the model will be built into the default Graph.
	with tf.Graph().as_default() as g:
		
		# Get images and flows for Flownet.
		img1, img2, flo = flownet_input.inputs(True, FLAGS.data_dir, FLAGS.batch_size)
		
		# Build a Graph that computes predictions from the inference model.
		logits = flowNet.inference(img1, img2, FLAGS.batch_size)
		
		# calculate EPE.
		x = flo
		y = tf.image.resize_images(logits[4], flowNet.get_size(x))
		flow2_loss = tf.reduce_mean(flowNet.compute_euclidean_distance(x,y))
		
		# Build the summary Tensor based on the TF collection of Summaries.
		summary_op = tf.summary.merge_all()

		# Create a saver to restore training checkpoints.
		saver = tf.train.Saver()

		# Instantiate a SummaryWriter to output summaries and the Graph.
		summary_writer = tf.summary.FileWriter(FLAGS.log_dir, g)

		while True:
			eval_once(saver, summary_writer, flow2_loss, x, y, summary_op)
			time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):  # pylint: disable=unused-argument
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	test()


if __name__ == '__main__':
	tf.app.run()
