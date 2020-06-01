#python train.py --tf_record_pattren=../lane_detect/tfrecords/train-?????-of-00002 --resnet_50_checkpoint_path=../lane_detect/res50/resnet_v2_50.ckpt --batch_size=2

import tensorflow as tf
import numpy as np
import os
import resnet as resnet
from TfRead import get_input
from scipy.misc import imsave

tf.flags.DEFINE_string("tf_record_pattren", "./tfrecords/train-?????-of-00002", "tfrecords path and pattern")
tf.flags.DEFINE_string("resnet_50_checkpoint_path", "./res50/resnet_v2_50.ckpt", "Resnet v2 checkpoint path")
tf.flags.DEFINE_string("batch_size", "16", "Batch size for training")
tf.flags.DEFINE_string("max_steps", "200000", "number of steps to train the network")
tf.flags.DEFINE_string("save_freq", "5000", "saving checkpoint and sample traiing results step value")
tf.flags.DEFINE_string("train_outs", "./chandu", "Trainng models and results saving path")
FLAGS = tf.flags.FLAGS
RESNET_V2 = 'resnet_v2_50'
RESNET_V2_CKPT_PATH = FLAGS.resnet_50_checkpoint_path
batch_size=int(FLAGS.batch_size)
HEIGHT = int(720/2.5)
WIDTH = int(1280/2.5)

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
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
      with tf.variable_scope("decoder_%d" % decoder_layer):
        if decoder_layer==0 or decoder_layer>=3:
          input = layers[-1]
        else:
          skip_layer = 4 - decoder_layer
          print(skip_layer)
          print(layers[-1], dict_1[str('block'+str(skip_layer))])
          input = tf.concat([layers[-1], dict_1[str('block'+str(skip_layer))]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, out_channels,'decoder_net_low')
        output = batch_norm(output, True)
        if dropout > 0.0 and True:
          output = tf.nn.dropout(output, keep_prob=1 - dropout)
        print('output'+str(decoder_layer), output)
        layers.append(output)
    with tf.variable_scope("decoder_last"):
      input = layers[-1] 
      rectified = tf.nn.relu(input)
      output = deconv(rectified, generator_outputs_channels,'decoder_net_low')
      output = tf.tanh(output)
      layers.append(output)
    return layers[-1]

def net_(xp, label_im, sess, is_train=True):
  global opt_non, gen_loss_mask_L1
  print('net__', xp, label_im)
  if True:    
    encoder_out, dict_1 = resnet.resnet(xp, RESNET_V2, is_train)
    decoder_out = decoder_net(encoder_out, 1, dict_1)
  with tf.variable_scope('L1_loss'):
    pred_im = decoder_out[:,:,:,-1]
    print(pred_im.shape,'predicted')
    L1_loss = tf.reduce_mean(tf.abs(pred_im - label_im))
    total_loss =  1* L1_loss 

  with tf.name_scope('optimizer') as scope:
    var = gen_tvars = [var for var in tf.trainable_variables() if not var.name.startswith("refine_network")]
    opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(total_loss  , var_list=var)
  global_step = tf.contrib.framework.get_or_create_global_step()
  incr_global_step = tf.assign(global_step, global_step+1)
  sess.run(tf.global_variables_initializer())
  return {  'total_loss':total_loss,
            'pred_im': pred_im,
            'train':tf.group(incr_global_step, opt),
            'global_step':global_step,
          }

def main(unused_argv):
    (input_im,label_im)=get_input(batch_size, HEIGHT, WIDTH, FLAGS.tf_record_pattren)
    m_folder = FLAGS.train_outs
    try: os.mkdir(m_folder)
    except: print("model directory already exists")
    try: os.mkdir(os.path.join(m_folder,'train_outs'))
    except: print("train_outs directory already exists")
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list= '0'
    coord = tf.train.Coordinator()
    sess = tf.Session(config=config)

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    network_out = net_(input_im, tf.squeeze(label_im), sess)


    fetches = {   "input_im": input_im,
                  "label_im":label_im,
                  "pred_im":network_out['pred_im'],
                  "loss":network_out['total_loss'],
                  'train': network_out['train'],
                  'global_step': network_out['global_step']
                }
    
    var_to_restore = tf.get_collection('pretrain')
    var = list(set(tf.global_variables()) - set(var_to_restore))
    restorer = tf.train.Saver(var_to_restore)
    print('loading checkpoint...')
    restorer.restore(sess, RESNET_V2_CKPT_PATH)
    print('checkpoint LOADED')
    # restorer = tf.train.Saver()
    # checkpoint_path = './hair_ckpt/model-56006'#model-90003'#'/media/DISK1/hair_teju/kk/train_outs_4k_08-22/model-31002'#'train_outs_4k_08-19/model-14000'
    # restorer.restore(sess, checkpoint_path)
    # print('checkpoint LOADED')
    # writer = tf.summary.FileWriter(m_folder+"/summary")
    # writer.add_graph(sess.graph)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    for i in range(int(FLAGS.max_steps)):
        final_ot = sess.run(fetches)
        print("Step:", final_ot["global_step"],  "Total loss:", final_ot["loss"])
        if i % int(FLAGS.save_freq) == 0:
          saver.save(sess, os.path.join(m_folder,"model") ,global_step=final_ot["global_step"])
          print('Saved model for the step'+str(i))
        if i % int(FLAGS.save_freq) == 0:
          imsave(os.path.join(m_folder,'train_outs',str(final_ot["global_step"])+'_label.png'), deprocess_image(np.squeeze(final_ot["label_im"][0])))
          imsave(os.path.join(m_folder,'train_outs',str(final_ot["global_step"])+'_input.png'), deprocess_image(np.squeeze(final_ot["input_im"][0])))
          imsave(os.path.join(m_folder,'train_outs',str(final_ot["global_step"])+'predicted.png'), deprocess_image(np.squeeze(final_ot["pred_im"][0])))
          print('Images saved for the step '+str(i))

if __name__ == '__main__':
    tf.app.run()
