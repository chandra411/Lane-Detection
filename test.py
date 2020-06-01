#python test.py --test_dir=../lane_detect/binary_lane_bdd/test/Images/ --checkpoint=../lane_detect/models/model-115003 --out_dir=./test_out
import tensorflow as tf
import numpy as np
import os
import resnet as resnet
import cv2
from scipy.misc import imsave
from scipy.misc import imread
# from sklearn.metrics import jaccard_similarity_score 
tf.flags.DEFINE_string("test_dir", "./test", "test images directory")
tf.flags.DEFINE_string("checkpoint", "./models/models-15001", "test images directory")
tf.flags.DEFINE_string("out_dir", "./test_out", "directory to save predicted test images ")
tf.flags.DEFINE_string("label_dir", "./label", "test images labels directory")
FLAGS = tf.flags.FLAGS
RESNET_V2 = 'resnet_v2_50'
HEIGHT = int(720 / 2.5)
WIDTH = int(1280 / 2.5)
batch_size = 1 
KEY = tf.GraphKeys.GLOBAL_VARIABLES

def lrelu(x, a=0.2):
  with tf.name_scope("lrelu"):
    x = tf.identity(x)
  return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def deprocess_image(image):
  image = image / 2.0 + 0.5
  return image

def deconv(batch_input, out_channels, _collection):
  with tf.variable_scope("deconv"):
    batch, in_height, in_width, in_channels = [
        int(d) for d in batch_input.get_shape()]
    filter = tf.get_variable("filter",
                             [4, 4, out_channels, in_channels],
                             dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0, 0.02),collections=[_collection, KEY])
    conv = tf.nn.conv2d_transpose(batch_input,
                          filter,
                          [batch, in_height * 2, in_width * 2, out_channels],
                          [1, 2, 2, 1],
                          padding="SAME")
    return conv

# seperate batch norm training and testing
def batch_norm(inputs, is_training=True, decay=0.999):
  with tf.variable_scope("batchnorm"):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    epsilon = 1e-5
    if is_training:
      batch_mean, batch_var = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=False)
      train_mean = tf.assign(pop_mean,
                             pop_mean * decay + batch_mean * (1 - decay))
      train_var = tf.assign(pop_var,
                            pop_var * decay + batch_var * (1 - decay))
      with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(inputs,
                                         batch_mean, batch_var, beta, scale, epsilon)
    else:
      return tf.nn.batch_normalization(inputs,
                                       pop_mean, pop_var, beta, scale, epsilon)


def decoder_net(encoder, generator_outputs_channels, dict_1):
    layer_specs = [(1024, 0.5), (512, 0.2),(256, 0.0), (64, 0.0)]

    layers = []
    layers.append(encoder)

    #num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
      #skip_layer = num_encoder_layers - decoder_layer - 1
      with tf.variable_scope("decoder_%d" % decoder_layer):
        if decoder_layer==0 or decoder_layer>=3:
          input = layers[-1]
        else:
          skip_layer = 4 - decoder_layer
          print(skip_layer)
          print(layers[-1], dict_1[str('block'+str(skip_layer))])
          input = tf.concat([layers[-1], dict_1[str('block'+str(skip_layer))]], axis=3)  #layers[-1] + dict_1['block'
        rectified = tf.nn.relu(input)
        
        output = deconv(rectified, out_channels,'decoder_net_low')
        output = batch_norm(output, True) #is_training=isTraining())

        if dropout > 0.0 and True: #isTraining():
          output = tf.nn.dropout(output, keep_prob=1 - dropout)
        print('output'+str(decoder_layer), output)
        layers.append(output)



    with tf.variable_scope("decoder_last"):
      input = layers[-1] #tf.concat([layers[-1], layers[0]], axis=3)
      rectified = tf.nn.relu(input)
      output = deconv(rectified, generator_outputs_channels,'decoder_net_low')
      output = tf.tanh(output)
      layers.append(output)

    return layers[-1]


def net_(xp, sess, is_train=True): 
  encoder_out, dict_1 = resnet.resnet(xp, RESNET_V2, is_train)
  decoder_out = decoder_net(encoder_out, 1, dict_1)
  pred_im = decoder_out[:,:,:,-1]
  return pred_im

def main(unused_argv):
    input_im = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 3])
    try: os.mkdir(FLAGS.out_dir)
    except: print("output directory already exists")
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list= '0'
    coord = tf.train.Coordinator()
    sess = tf.Session(config=config)

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    out_img = net_(input_im, sess)
    restorer = tf.train.Saver()
    restorer.restore(sess, FLAGS.checkpoint)
    print('checkpoint LOADED')
    # test_path = './binary_lane_bdd/Images/'
    imgs = [j for j in os.listdir(FLAGS.test_dir) if j.endswith('.jpg')]
    ious = np.zeros((len(imgs)))
    for i in range(0,len(imgs)):
        img = imread(os.path.join(FLAGS.test_dir,imgs[i]))
        h, w, c = img.shape
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = (img.astype(np.float32)/255.0)
        img = (img -0.5) *2 
        out = sess.run(out_img, feed_dict={input_im: np.expand_dims(img, axis=0)})
        out = (out / 2.0 + 0.5)*255
        out = out[0].astype(np.uint8)
        out = cv2.resize(out, (w,h))
        imsave(os.path.join(FLAGS.out_dir,imgs[i]), out)
        print(imgs[i]+"  saved into " + FLAGS.out_dir)
    #     lab = imread(os.path.join(FLAGS.label_dir,imgs[i]))
    #     out[out<=100]=0
    #     out[out>100]=1
    #     lab[out<=100]=0
    #     lab[out>100]=1
    #     iou = jaccard_similarity_score(out, lab[:,:,0])
    #     ious[i] = iou 
    #     print(imgs[i]+"- IOU is:: ", iou)
    # print("Toatal IOU is :: ", np.mean(ious))
if __name__ == '__main__':
    tf.app.run()
