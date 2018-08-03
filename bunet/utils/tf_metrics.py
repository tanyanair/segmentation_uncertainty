"""
Tensorflow implementation of metrics and utils
"""
import tensorflow as tf

EPSILON = 1e-5
_smooth = 1


def dice_coef(y_tru, y_prd):
    y_tru = tf.reshape(y_tru, [2, -1])
    y_prd = tf.reshape(y_prd, [2, -1])
    y_prd_pr = tf.sigmoid(y_prd)
    intersection = tf.reduce_sum(y_prd_pr * y_tru, 0)
    union = tf.reduce_sum(y_prd_pr, 0) + tf.reduce_sum(y_tru, 0)
    dice = (2. * intersection + _smooth) / (union + _smooth)
    return tf.reduce_mean(dice)

def dice_coef_loss(y_tru, y_prd):
    return -dice_coef(y_tru, y_prd)

def weighted_mc_xentropy(y_tru, mu, log_var, nb_mc, weight, batch_size):
    mu_mc = tf.tile(mu, [1, 1, 1, 1, nb_mc])
    std = tf.exp(log_var)
    # hard coded the known shape of the data
    noise = tf.random_normal((batch_size, 192, 192, 64, nb_mc)) * std
    prd = mu_mc + noise

    y_tru = tf.tile(y_tru, [1, 1, 1, 1, nb_mc])
    mc_x = tf.nn.weighted_cross_entropy_with_logits(targets=y_tru, logits=prd, pos_weight=weight)
    # mean across mc samples
    mc_x = tf.reduce_mean(mc_x, -1)
    # mean across every thing else
    return tf.reduce_mean(mc_x)

