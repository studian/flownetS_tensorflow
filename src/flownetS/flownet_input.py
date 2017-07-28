from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf

IMAGE_HEIGHT = 384#384, 436
IMAGE_WIDTH = 512#512, 1024
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 22232
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 640#640, 1041
def read_flo(filename_queue):
	value = tf.read_file(filename_queue)
	record_4bytes = tf.decode_raw(value, tf.float32)
	result = tf.reshape(tf.strided_slice(record_4bytes, [3],[3 + 2 * IMAGE_HEIGHT * IMAGE_WIDTH]),[IMAGE_HEIGHT, IMAGE_WIDTH, 2])
	return result

def read_labeled_image_list(eval_data, data_dir):
	"""Reads a .txt file containing pathes and labeles
	Args:
		 data_dir : list dir
	Returns:
		 List with all filenames in file image_list_file
	"""
	image0_list = []
	image1_list = []
	flo_list = []

	#Read list
	if eval_data:
		fimg1 = open(os.path.join(data_dir, 'img1_list_test.txt'), 'r')
		fimg2 = open(os.path.join(data_dir, 'img2_list_test.txt'), 'r')
		fflo = open(os.path.join(data_dir, 'flo_list_test.txt'), 'r')
	else:
		fimg1 = open(os.path.join(data_dir, 'img1_list_train.txt'), 'r')
		fimg2 = open(os.path.join(data_dir, 'img2_list_train.txt'), 'r')
		fflo = open(os.path.join(data_dir, 'flo_list_train.txt'), 'r')

	for filename in fimg1:
		if not tf.gfile.Exists(filename[:-1]):
			raise ValueError('Failed to find file: ' + filename[:-1])
		image0_list.append(filename[:-1])
	
	for filename in fimg2:
		if not tf.gfile.Exists(filename[:-1]):
			raise ValueError('Failed to find file: ' + filename[:-1])
		image1_list.append(filename[:-1])

	for filename in fflo:
		if not tf.gfile.Exists(filename[:-1]):
			raise ValueError('Failed to find file: ' + filename[:-1])
		flo_list.append(filename[:-1])

	fflo.close()
	fimg1.close()
	fimg2.close()

	return image0_list, image1_list, flo_list

def read_images_from_disk(input_queue):
	"""Consumes a single filename and label as a ' '-delimited string.
	Args:
		filename_tensor: A scalar string tensor.
	Returns:
		Three tensors: the decoded images and flos.
	"""
	image0_file = tf.read_file(input_queue[0])
	image0 = tf.image.decode_image(image0_file)
	image1_file = tf.read_file(input_queue[1])
	image1 = tf.image.decode_image(image1_file)
	flo = read_flo(input_queue[2])

	return image0, image1, flo

def _generate_image_and_label_batch(image0, image1, flo, min_queue_examples, batch_size, shuffle):
	"""Construct a queued batch of images and labels.

	Args:
		image1, image2: 3-D Tensor of [height, width, 3] of type.float32.
		min_queue_examples: int32, minimum number of samples to retain
			in the queue that provides of batches of examples.
		batch_size: Number of images per batch.
		shuffle: boolean indicating whether to use a shuffling queue.

	Returns:
		bimages: Images. 4D tensor of [batch_size, height, width, 3] size.
		bflo: Flos. 4D tensor of [batch_size, height, width, 2] size.
	"""
	# Create a queue that shuffles the examples, and then
	# read 'batch_size' images + labels from the example queue.
	num_preprocess_threads = 16
	if shuffle:
		bimage0, bimage1, bflo = tf.train.shuffle_batch(
				[image0, image1, flo],
				batch_size=batch_size,
				num_threads=num_preprocess_threads,
				capacity=min_queue_examples + 3 * batch_size,
				min_after_dequeue=min_queue_examples)
	else:
		bimage0, bimage1, bflo = tf.train.batch(
				[image0, image1, flo],
				batch_size=batch_size,
				num_threads=num_preprocess_threads,
				capacity=min_queue_examples + 3 * batch_size)

	# Display the training images in the visualizer.
	tf.summary.image('images1', bimage0)
	tf.summary.image('images2', bimage1)
	return bimage0, bimage1, bflo

def inputs(eval_data, data_dir, batch_size):
	"""Construct input for CIFAR evaluation using the Reader ops.

	Args:
		eval_data: bool, indicating if one should use the train or eval data set.
		data_dir: Path to the FlowNet data directory.
		batch_size: Number of images per batch.

	Returns:
		image1, image2: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		flo: Flows. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 2] size.
	"""
	image1_list, image2_list, flo_list = read_labeled_image_list(eval_data, data_dir)

	images1 = tf.convert_to_tensor(image1_list, dtype=tf.string)
	images2 = tf.convert_to_tensor(image2_list, dtype=tf.string)
	flos = tf.convert_to_tensor(flo_list, dtype=tf.string)

	input_queue = tf.train.slice_input_producer([images1, images2, flos])
	image1, image2, flo = read_images_from_disk(input_queue)
	if eval_data:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
	else:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	height = IMAGE_HEIGHT
	width = IMAGE_WIDTH

	image1 = tf.cast(image1, tf.float32)
	image2 = tf.cast(image2, tf.float32)
	image1.set_shape([height, width, 3])
	image2.set_shape([height, width, 3])

	# Because these operations are not commutative, consider randomizing the order their operation.

	if eval_data:
		distorted_image1 = tf.image.per_image_standardization(image1)
		distorted_image2 = tf.image.per_image_standardization(image2)

	else:
		#distorted_image1 = tf.image.adjust_gamma(image1, gamma = random.uniform(0.7, 1.5))
		#distorted_image2 = tf.image.adjust_gamma(image2, gamma = random.uniform(0.7, 1.5))

		distorted_image1 = tf.image.random_brightness(image1, max_delta=255/4.0, seed=1)
		distorted_image2 = tf.image.random_brightness(image2, max_delta=255/4.0, seed=1)
		distorted_image1 = tf.image.random_contrast(distorted_image1, lower=0.2, upper=1.8, seed=2)
		distorted_image2 = tf.image.random_contrast(distorted_image2, lower=0.2, upper=1.8, seed=2)

		distorted_image1 = tf.image.per_image_standardization(distorted_image1)
		distorted_image2 = tf.image.per_image_standardization(distorted_image2)

	flo.set_shape([height, width, 2])
	#distorted_image1 = tf.image.resize_images(distorted_image1, [384, 512]) 
	#distorted_image2 = tf.image.resize_images(distorted_image2, [384, 512])
	#flo = tf.image.resize_images(flo, [384, 512])
	min_queue_examples = 10 * batch_size
	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(distorted_image1, distorted_image2, flo, min_queue_examples, batch_size, shuffle=not eval_data)
