import tensorflow as tf


def weight_variable(shape):
    return tf.get_variable(name='W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def weight_variable_devonc(shape):
    return tf.get_variable(name='W', shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    return tf.Variable(tf.constant(0., shape=shape))


def conv3d(x, w, name, s=1, pd='SAME'):
    cnv = tf.nn.convolution(x, w,  padding=pd, strides=[s, s, s], name=name)
    return cnv


def deconv3d(x, w, name, s=1, pd='SAME'):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]*2, x_shape[4]])
    dcnv = tf.nn.conv3d_transpose(x, w, output_shape, strides=[1, s, s, s, 1], padding=pd, name=name)
    return dcnv


def conv(x, k, in_ch, out_ch, stride, dr, name, activation=None):
    """
    :param x: input tensor
    :param int k: kernel size
    :param int in_ch: number of input channels
    :param int out_ch: number of output channels
    :param int stride: stride of convolution
    :param float dr: dropout rate (probability of keeping the node)
    :param str name: name applied to layer
    :param activation: activation function
    :return:
    """
    with tf.variable_scope("conv_layer{}".format(name)):
        w1 = weight_variable([k, k, k, in_ch, out_ch])
        b1 = bias_variable([out_ch])
        net = conv3d(x, w1, 'c_{}'.format(stride), s=stride) + b1
        if activation is not None:
            net = activation(net)
        if dr is not None:
            net = tf.nn.dropout(net, dr)
    return net, w1, b1


def deconv(x, k, in_ch, out_ch, stride, dr, name, activation=None):
    with tf.variable_scope("deconv_layer{}".format(name)):
        w1 = weight_variable([k, k, k, in_ch, out_ch])
        b1 = bias_variable([out_ch])
        net = deconv3d(x, w1, 'd_{}'.format(stride), s=stride) + b1
        if activation is not None:
            net = activation(net)
        if dr is not None:
            net = tf.nn.dropout(net, dr)
    return net, w1, b1

