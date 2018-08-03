import numpy as np
import logging

import tensorflow as tf
from bunet.models.layers import conv, deconv
from bunet.utils.tf_metrics import dice_coef, weighted_mc_xentropy

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
EPSILON = 1e-5


def build_network(x, dr, nb_ch, nb_kers=32):
    """
    :param x: input tensor, shape [batch_size,nx,ny,nz,channels]
    :param dr: dropout probability tensor
    :param nb_ch: number of input channels
    :param nb_kers: number of features in the first layer
    :return: network and dict of variables
    """
    logging.info("features {nb_feat}, filter size 3x3, pool size: 2x2".format(nb_feat=nb_kers))

    convs = []
    variables = []

    c1, w1, b1 = conv(x, 3, nb_ch, nb_kers, stride=1, dr=None, name='c1', activation=tf.nn.relu)
    c1p, w1p, b1p = conv(x, 3, nb_ch, nb_kers, stride=2, dr=None, name='c1p', activation=tf.nn.relu)
    c2, w2, b2 = conv(c1p, 3, nb_kers, nb_kers * 2, stride=1, dr=None, name='c2', activation=tf.nn.relu)
    c2p, w2p, b2p = conv(c2, 3, nb_kers * 2, nb_kers * 2, stride=2, dr=None, name='c2p', activation=tf.nn.relu)
    c3, w3, b3 = conv(c2p, 3, nb_kers * 2, nb_kers * 4, stride=1, dr=None, name='c3', activation=tf.nn.relu)
    c3p, w3p, b3p = conv(c3, 3, nb_kers * 4, nb_kers * 4, stride=2, dr=dr, name='c3p', activation=tf.nn.relu)
    c4, w4, b4 = conv(c3p, 3, nb_kers * 4, nb_kers * 8, stride=1, dr=None, name='c4', activation=tf.nn.relu)
    c4p, w4p, b4p = conv(c4, 3, nb_kers * 8, nb_kers * 8, stride=2, dr=dr, name='c4p', activation=tf.nn.relu)
    d4, dw4, db4 = deconv(c4p, 3, nb_kers * 8, nb_kers * 8, 2, None, 'd4', tf.nn.relu)
    d4c, dw4c, db4c = conv(tf.add(d4, c4, 'add_d4_c4'), 3, nb_kers * 8, nb_kers * 4, 1, dr, 'd4c', tf.nn.relu)
    d3, dw3, db3 = deconv(d4c, 3, nb_kers * 4, nb_kers * 4, 2, None, 'd3', tf.nn.relu)
    d3c, dw3c, db3c = conv(tf.add(d3, c3, 'add_d3_c3'), 3, nb_kers * 4, nb_kers * 2, 1, dr, 'd3c', tf.nn.relu)
    d2, dw2, db2 = deconv(d3c, 3, nb_kers * 2, nb_kers * 2, 2, None, 'd2', tf.nn.relu)
    d2c, dw2c, db2c = conv(tf.add(d2, c2, 'add_d2_c2'), 3, nb_kers * 2, nb_kers, 1, None, 'd2c', tf.nn.relu)
    d1, dw1, db1 = deconv(d2c, 3, nb_kers, nb_kers, 2, dr, 'd1', tf.nn.relu)
    d1c, dw1c, db1c = conv(tf.add(d1, c1), 1, nb_kers, 1, 1, None, 'mu', activation=None)
    d1c_2, dw1c_2, db1c_2 = conv(tf.add(d1, c1), 1, nb_kers, 1, 1, None, 'log_var', activation=None)

    variables.extend([w1p, w2, w2p, w3, w3p, w4, w4p,
                      dw4, dw4c, dw3, dw3c, dw2, dw2c, dw1, dw1c, dw1c_2,
                      b1p, b2, b2p, b3, b3p, b4, b4p,
                      db4, db4c, db3, db3c, db2, db2c, db1, db1c, db1c_2])
    convs.extend([c1p, c2, c2p, c3, c3p, c4, c4p,
                  d4, d4c, d3, d3c, d2, d2c, d1, d1c, d1c_2])
    print("==========================================\n     Layer Summary")
    for i, c in enumerate(convs):
        print('     Layer  {}  Output Shape  {}'.format(i, c.get_shape()))

    return d1c, d1c_2, variables


class BUnet(object):
    def __init__(self, nb_ch=3, nb_kers=32, nb_mc=5, weight_decay=None, batch_size=3):
        tf.reset_default_graph()

        self.x = tf.placeholder(tf.float32, shape=[None, None, None, None, nb_ch])
        self.y = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        self.class_weight = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
        self.batch_size = batch_size

        mu, log_var, self.variables = build_network(self.x, self.keep_prob, nb_ch, nb_kers)
        nb_trainable_params = np.sum([np.prod(v.shape) for v in self.variables])
        logging.info('Number of trainable parameters: {}'.format(nb_trainable_params))

        loss = weighted_mc_xentropy(self.y, mu, log_var, nb_mc, self.class_weight, self.batch_size)
        if weight_decay is not None:
            l2_loss = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss += weight_decay * l2_loss
        self.loss = loss
        self.predictor = mu
        self.log_variance = log_var
        self.dice = dice_coef(self.y, mu)
        self.prd_min = tf.reduce_min(tf.sigmoid(mu))
        self.prd_max = tf.reduce_max(tf.sigmoid(mu))
        self.ppv = tf.metrics.precision(labels=self.y, predictions=tf.sigmoid(mu))
        self.tpr = tf.metrics.recall(labels=self.y, predictions=tf.sigmoid(mu))

    @staticmethod
    def save(sess, model_path):
        """
        Saves the current session to a checkpoint
        :param sess: current session
        :param model_path: path to file system location
        """
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    @staticmethod
    def restore(sess, model_path):
        """
        Restores a session from a checkpoint
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)
