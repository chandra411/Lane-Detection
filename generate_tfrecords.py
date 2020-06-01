#python generate_tfrecords.py --image_dir={path to your input images} --label_dir={path to your label images} --tfr_dir={path to save tf records}

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
from scipy.misc import imresize

import json
import os.path
import random
import sys
import threading

import numpy as np
import pickle as pkl
import scipy.io as sio
import tensorflow as tf

tf.flags.DEFINE_string("image_dir", "/content/drive/My Drive/Lane_Detection/data/Images/",
											 "Training images data directory")
tf.flags.DEFINE_string("label_dir", "/content/drive/My Drive/Lane_Detection/data/Labels/",
											 "Training labes data directory")

tf.flags.DEFINE_string("tfr_dir", "/content/drive/My Drive/Lane_Detection/data/tfrecords_aug", "Output data directory.")

tf.flags.DEFINE_string("prefix", "", "Prefix of the tensorflow record files.")

tf.flags.DEFINE_integer("train_shards", 2,
												"Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("num_threads", 2,
												"Number of threads to preprocess the images.")
FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",["model","mask"])


class ImageDecoder(object):
	"""Helper class for decoding images in TensorFlow."""

	def __init__(self):
		# Create a single TensorFlow Session for all image decoding calls.
		config = tf.ConfigProto()
		config.gpu_options.visible_device_list= '1'
		self._sess = tf.Session(config=config)

		# TensorFlow ops for JPEG decoding.
		self._encoded_jpeg = tf.placeholder(dtype=tf.string)
		self._encoded_png = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)
		self._decode_png = tf.image.decode_png(self._encoded_png, channels=3)

		self._decoded_jpeg = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])
		self._encode_jpeg = tf.image.encode_jpeg(self._decoded_jpeg)

	def decode_jpeg(self, encoded_jpeg):
		image = self._sess.run(self._decode_jpeg,feed_dict={self._encoded_jpeg: encoded_jpeg})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image

	def decode_png(self, encoded_png):
		image = self._sess.run(self._decode_png,
													 feed_dict={self._encoded_png: encoded_png})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image

	def encode_jpeg(self, decoded_image):
		image = self._sess.run(self._encode_jpeg,
													 feed_dict={self._decoded_jpeg: decoded_image})
		return image

def _int64_feature(value):
	"""Wrapper for inserting an int64 Feature into a SequenceExample proto."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	"""Wrapper for inserting a bytes Feature into a SequenceExample proto."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _float_feature(values):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[a for a in values]))

def _int64_feature_list(values):
	"""Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
	return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
	"""Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
	return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _to_tf_example(image, decoder):
	"""Builds a TF Example proto for an image triplet.

	Args:
		image: An ImageMetadata object.
		decoder: An ImageDecoder object.

	Returns:
		A TF Example proto.
	"""
	with open(FLAGS.image_dir + image.model, "r") as f:
		encoded_model = f.read()
		        
	with open(FLAGS.label_dir + image.mask, "r") as f:
		encoded_mask = f.read()
			
	try:
		decoded_model = decoder.decode_jpeg(encoded_model)
		decoded_mask = decoder.decode_jpeg(encoded_mask)
		
	except (tf.errors.InvalidArgumentError, AssertionError):
		print("Skipping file with invalid JPEG data: %s" % image.image_id)
		print("Skipping file with invalid JPEG data: %s" % image.product_image_id)
		return
	tf_example = tf.train.Example(features=tf.train.Features(feature={
				'model': _bytes_feature(encoded_model),
				'mask': _bytes_feature(encoded_mask),
				}))

	return tf_example


def _process_image_files(thread_index, ranges, name, images, decoder,num_shards):

	"""Processes and saves a subset of images as TFRecord files in one thread.

	Args:
		thread_index: Integer thread identifier within [0, len(ranges)].
		ranges: A list of pairs of integers specifying the ranges of the dataset to
			process in parallel.
		name: Unique identifier specifying the dataset.
		images: List of ImageMetadata.
		decoder: An ImageDecoder object.
		num_shards: Integer number of shards for the output files.
	"""
	# Each thread produces N shards where N = num_shards / num_threads. For
	# instance, if num_shards = 128, and num_threads = 2, then the first thread
	# would produce shards [0, 64).
	num_threads = len(ranges)
	assert not num_shards % num_threads
	num_shards_per_batch = int(num_shards / num_threads)

	shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
														 num_shards_per_batch + 1).astype(int)
	num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

	counter = 0
	for s in xrange(num_shards_per_batch):
		# Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
		shard = thread_index * num_shards_per_batch + s
		output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
		output_file = os.path.join(FLAGS.tfr_dir, output_filename)
		writer = tf.python_io.TFRecordWriter(output_file)

		shard_counter = 0
		images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
		for i in images_in_shard:
			image = images[i]
			tf_example = _to_tf_example(image, decoder)
			if tf_example is not None:
				writer.write(tf_example.SerializeToString())
				shard_counter += 1
				counter += 1

			if not counter % 1000:
				sys.stdout.flush()

		writer.close()
		sys.stdout.flush()
		shard_counter = 0
	sys.stdout.flush()


def _process_dataset(name, images, num_shards):
	"""Processes a complete data set and saves it as a TFRecord.

	Args:
		name: Unique identifier specifying the dataset.
		images: List of ImageMetadata.
		num_shards: Integer number of shards for the output files.
	"""
	
	# Shuffle the ordering of images. Make the randomization repeatable.
	random.seed(12345)
	random.shuffle(images)

	# Break the images into num_threads batches. Batch i is defined as
	# images[ranges[i][0]:ranges[i][1]].
	num_threads = min(num_shards, FLAGS.num_threads)
	spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
	ranges = []
	threads = []
	for i in xrange(len(spacing) - 1):
		ranges.append([spacing[i], spacing[i + 1]])

	# Create a mechanism for monitoring when all threads are finished.
	coord = tf.train.Coordinator()

	# Create a utility for decoding JPEG images to run sanity checks.
	decoder = ImageDecoder()

	# Launch a thread for each batch.
	print("Launching %d threads for spacings: %s" % (num_threads, ranges))
	for thread_index in xrange(len(ranges)):
		args = (thread_index, ranges, name, images, decoder, num_shards)
		t = threading.Thread(target=_process_image_files, args=args)
		t.start()
		threads.append(t)

	# Wait for all the threads to terminate.
	coord.join(threads)
	print("%s: Finished processing all %d image pairs in data set '%s'." %
				(datetime.now(), len(images), name))



def _load_and_process_metadata():
	"""
	Returns:
		A list of ImageMetadata.
	"""
	image_list = [f for f in os.listdir(FLAGS.image_dir) if f.endswith('.jpg')]
	image_metadata = []
	
	for item in image_list:
		image_pair = [item, item]
		print(image_pair)
		image_metadata.append(ImageMetadata(image_pair[0], image_pair[1]))
		
	print("Finished processing %d pairs for %d images in" % 
				(len(image_pair), len(image_pair)))
				

	return image_metadata


def main(unused_argv):
	def _is_valid_num_shards(num_shards):
		"""Returns True if num_shards is compatible with FLAGS.num_threads."""
		return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

	assert _is_valid_num_shards(FLAGS.train_shards), (
			"Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")

	if not tf.gfile.IsDirectory(FLAGS.tfr_dir):
		tf.gfile.MakeDirs(FLAGS.tfr_dir)

	
	# Load image metadata from label files.
	train_dataset = _load_and_process_metadata()
	
	print(len(train_dataset))

	_process_dataset(FLAGS.prefix + "train", train_dataset, FLAGS.train_shards)

if __name__ == "__main__":
	tf.app.run()
