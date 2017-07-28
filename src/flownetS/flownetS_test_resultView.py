from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import struct
import cv2

import numpy as np
import tensorflow as tf
import flownet_S as flowNet
import flownet_input 
import time
import math
from datetime import datetime
from lib import flowlib
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './test_log/flownetS', """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', './data', """Directory where to get data list.""")
tf.app.flags.DEFINE_integer('batch_size', 8, """Batch size.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './train_weight/flownetS', """Directory where to read model checkpoints.""")
#tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, """How often to run the eval.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 30, """How often to run the eval.""")

class Flow(object):
    def __init__(self, floname):
        self.floname = floname

    def read_flo(self):
            with open(self.floname, "rb") as f:
                data = f.read()
            self.width = struct.unpack('@i', data[4:8])[0]
            self.height = struct.unpack('@i', data[8:12])[0]
            self.flodata = np.zeros((self.height, self.width, 2))
            for i in range(self.width*self.height):
                data_u = struct.unpack('@f', data[12+8*i:16+8*i])[0]
                data_v = struct.unpack('@f', data[16+8*i:20+8*i])[0]
                n = int(i / self.width)
                k = np.mod(i, self.width)
                self.flodata[n, k, :] = [data_u, data_v]
            return self.flodata

    def get_colorwheel(self):
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6
        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros([ncols, 3])

        col = 0
        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.floor([x * (255/RY) for x in range(RY)])
        col = col + RY
        # YG
        colorwheel[col:col+YG, 0] = 255 - np.floor([x * (255/YG) for x in range(YG)])
        colorwheel[col:col+YG, 1] = 255
        col = col + YG
        # GC
        colorwheel[col:col+GC, 1] = 255
        colorwheel[col:col+GC, 2] = np.floor([x * (255/GC) for x in range(GC)])
        col = col + GC
        # CB
        colorwheel[col:col+CB, 1] = 255 - np.floor([x * (255/CB) for x in range(CB)])
        colorwheel[col:col+CB, 2] = 255
        col = col + CB
        # BM
        colorwheel[col:col+BM, 2] = 255
        colorwheel[col:col+BM, 0] = np.floor([x * (255/BM) for x in range(BM)])
        col = col + BM
        # MR
        colorwheel[col:col+MR, 2] = 255 - np.floor([x * (255/MR) for x in range(MR)])
        colorwheel[col:col+MR, 0] = 255
        return colorwheel

    def print_flo(self):
        self.flodata = self.read_flo()
        u = self.flodata[:, :, 0]
        v = self.flodata[:, :, 1]
        img = np.zeros([self.height, self.width, 3])
        # normalization
        rad = np.amax((u ** 2 + v ** 2) ** 0.5)
        eps = np.finfo(float).eps
        u = u / (rad + eps)
        v = v / (rad + eps)
        # image a colorwheel, if we have arc length and radius, it's easy to locate an exact color
        colorwheel = self.get_colorwheel()
        rad = (u ** 2 + v ** 2) ** 0.5
        arc = np.arctan2(-v, -u) / np.pi
        # the number of color's level in which R/G/B channel
        ncols = colorwheel.shape[0]
        # [-1, 1] maped to [1, ncols]
        level = (arc+1) / 2 * (ncols-1) + 1
        level = level.reshape((-1, 1))
        level_floor = [int(x) for x in level]
        level_ceil = [x+1 for x in level_floor]
        for x in level_ceil:
            if x == ncols + 1:
                x = 1
        mask = list(map(lambda x: x[0]-x[1], zip(level, level_floor)))
        for i in range(colorwheel.shape[1]):
            tmp = colorwheel[:, i]
            tmp = list(tmp)
            col0 = []
            col1 = []
            for x in level_floor:
                col0.append(tmp[x-1])

            for x in level_ceil:
                col1.append(tmp[x-1])
            # transfer to matrix for compute
            mask = np.array(mask).reshape((self.height, self.width))
            col0 = np.array(col0).reshape((self.height, self.width))
            col0 = col0 / 255.
            col1 = np.array(col1).reshape((self.height, self.width))
            col1 = col1 / 255.
            col = (1-mask) * col0 + mask * col1
            # dont konw why need follow code,
            # increase saturation with radius?
            col = col.reshape((-1, 1))
            rad = rad.reshape((-1, 1))
            m = 0
            for x in rad:
                if x <= 1:
                    col[m][0] = 1 - x * (1 - col[m][0])
                    col[m][0] = int(255 * col[m][0])
                    m = m + 1
                else:
                    col[m][0] = col[m][0] * 0.75
                    col[m][0] = int(255 * col[m][0])
                    m = m + 1
            col = col.reshape((self.height, self.width))
            img[:, :, i] = col
        return img

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
		
		cv2.namedWindow('pimage')
		cv2.namedWindow('gtimage')

		while True:
			eval_once(saver, summary_writer, flow2_loss, x, y, summary_op)

			pflo_file = Flow('p.flo').read_flo()
			pimage = Flow('p.flo').print_flo()

			gtflo_file = Flow('gt.flo').read_flo()
			gtimage = Flow('gt.flo').print_flo()

			cv2.imshow('pimage', pimage.astype(np.uint8))
			cv2.imwrite('p.png', pimage)

			cv2.imshow('gtimage', gtimage.astype(np.uint8))
			cv2.imwrite('gt.png', gtimage)

			c = cv2.waitKey(10)
			if c == 27:
				break
		cv2.desttroyAllWindows()
			#time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):  # pylint: disable=unused-argument
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)		

	tf.gfile.MakeDirs(FLAGS.log_dir)
	test()


if __name__ == '__main__':
	tf.app.run()
