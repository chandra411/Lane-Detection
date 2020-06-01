from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

# input_file_pattern = "./tfrecords/train-?????-of-00002"
values_per_input_shard = 443
num_preprocess_threads = 2
def process_image(encoded_model,
					encoded_mask,
					is_training=True,
					height=288,
					width=512,
					resize_height=288,
					resize_width=512,
					thread_id=0,
					image_format="jpeg",
					zero_one_mask=True,
					different_image_size=False):
	
	
	def image_summary(name, image):
		return
		
	# Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
	with tf.name_scope("decode", values=[encoded_model]):
		if image_format == "jpeg":
			model = tf.image.decode_jpeg(encoded_model, channels=3)
			mask = tf.image.decode_jpeg(encoded_mask, channels=1)
		elif image_format == "png":
			model = tf.image.decode_png(encoded_model, channels=3)
			mask = tf.image.decode_png(encoded_mask, channels=1)
		else:
			raise ValueError("Invalid image format: %s" % image_format)

	model = tf.image.convert_image_dtype(model, dtype=tf.float32)
	mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
	image_summary("original_image", model)
	image_summary("original_human", mask)
	
	assert (resize_height > 0) == (resize_width > 0)
	if different_image_size:
		model = tf.image.resize_images(model,size=[height, width],method=tf.image.ResizeMethod.BILINEAR)
		mask = tf.image.resize_images(mask,size=[height, width],method=tf.image.ResizeMethod.BILINEAR)
	else:
		model = tf.image.resize_images(model,size=[resize_height, resize_width],method=tf.image.ResizeMethod.BILINEAR)
		mask = tf.image.resize_images(mask,size=[resize_height, resize_width],method=tf.image.ResizeMethod.BILINEAR)
	
	
	image_summary("final_image", model)
	image_summary("final_human", mask)
#	image_summary("final_prod_image", prod_image)
	
	#Rescale to [-1,1] instead of [0, 1]
	model = (model - 0.5) * 2.0
	mask = (mask - 0.5) * 2.0
	return model,mask#, prod_image



def parse_tf_example(serialized, stage=""):
	features = tf.parse_single_example(
			serialized,
			features={
					"model": tf.FixedLenFeature([], tf.string),
					"mask": tf.FixedLenFeature([], tf.string)
			}
	)
	encoded_model = features["model"]
	encoded_mask = features["mask"]

	return (encoded_model, encoded_mask)
	
def prefetch_input_data(reader,
						file_pattern,
						is_training,
						batch_size,
						values_per_shard,
						input_queue_capacity_factor=16,
						num_reader_threads=1,
						shard_queue_name="filename_queue",
						value_queue_name="input_queue"):
	data_files = []
	for pattern in file_pattern.split(","):
		data_files.extend(tf.gfile.Glob(pattern))
	if not data_files:
		tf.logging.fatal("Found no input files matching %s", file_pattern)
	else:
		tf.logging.info("Prefetching values from %d files matching %s",
										len(data_files), file_pattern)

	if is_training:
		filename_queue = tf.train.string_input_producer(
				data_files, shuffle=True, capacity=16, name=shard_queue_name)
		min_queue_examples = values_per_shard * input_queue_capacity_factor
		capacity = min_queue_examples + 100 * batch_size
		values_queue = tf.RandomShuffleQueue(
				capacity=capacity,
				min_after_dequeue=min_queue_examples,
				dtypes=[tf.string],
				name="random_" + value_queue_name)
	else:
		filename_queue = tf.train.string_input_producer(
				data_files, shuffle=False, capacity=1, name=shard_queue_name)
		capacity = values_per_shard + 3 * batch_size
		values_queue = tf.FIFOQueue(
				capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

	enqueue_ops = []
	for _ in range(num_reader_threads):
		_, value = reader.read(filename_queue)
		enqueue_ops.append(values_queue.enqueue([value]))
	tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
			values_queue, enqueue_ops))
	tf.summary.scalar(
			"queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
			tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

	return values_queue
def get_input(batch_size, H, W, input_file_pattern):
	# Load input data
	input_queue = prefetch_input_data(
			tf.TFRecordReader(),
			input_file_pattern,
			is_training=True,
			batch_size=batch_size,
			values_per_shard=values_per_input_shard,
			input_queue_capacity_factor=2,
			num_reader_threads=num_preprocess_threads)

	# Image processing and random distortion. Split across multiple threads
	images_and_maps = []

	for thread_id in range(num_preprocess_threads):
		serialized_example = input_queue.dequeue()
		(encoded_model, encoded_mask) = parse_tf_example(serialized_example)

		#Body Segment is Human now
		(model_image, mask_image) = process_image(encoded_model, encoded_mask,
						height=H, width=W, resize_height=H, resize_width=W)

		images_and_maps.append([model_image,mask_image])

	# Batch inputs.
	queue_capacity = (7 * num_preprocess_threads *batch_size)

	return tf.train.batch_join(images_and_maps,batch_size=batch_size,capacity=queue_capacity,name="batch")
