"""
Tensorflow implementation of metrics and utils
"""
import tensorflow as tf
import numpy as np

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


def hard_dice(y_tru, y_prd):
    y_prd = tf.sigmoid(y_prd)
    y_prd = tf.round(y_prd)
    intersection = tf.reduce_sum(y_prd * y_tru)
    union = tf.reduce_sum(y_prd) + tf.reduce_sum(y_tru)
    dice = (2. * intersection + _smooth) / (union + _smooth)
    return dice


def generalized_dice_loss(y_tru, y_prd):
    y_prd_pr = tf.sigmoid(y_prd)

    # w1 = 1. / tf.square((tf.reduce_sum(y_tru)))
    # w0 = 1. / tf.square((tf.reduce_sum(1. - y_tru)))
    w1 = 8
    w0 = 1

    numerator = w1 * tf.reduce_sum(y_tru * y_prd_pr) + \
                w0 * tf.reduce_sum((1. - y_tru) * (1. - y_prd_pr))

    denominator = w1 * tf.reduce_sum(y_tru * y_prd_pr) + \
                  w0 * tf.reduce_sum((1. - y_tru) + (1. - y_prd_pr))

    gdl = 1 - 2 * numerator / denominator
    return gdl


def weighted_mc_xentropy(y_tru, mu, log_var, nb_mc, weight, batch_size):
    mu_mc = tf.tile(mu, [1, 1, 1, 1, nb_mc])
    std = tf.exp(log_var)
    noise = tf.random_normal((batch_size, 192, 192, 64, nb_mc)) * std
    prd = mu_mc + noise

    y_tru = tf.tile(y_tru, [1, 1, 1, 1, nb_mc])
    mc_x = tf.nn.weighted_cross_entropy_with_logits(targets=y_tru, logits=prd, pos_weight=weight)
    # mean across mc samples
    mc_x = tf.reduce_mean(mc_x, -1)
    # mean across every thing else
    return tf.reduce_mean(mc_x)


def cross_entropy(y_tru, y_prd, class_weight):
    flat_prd = tf.reshape(y_prd, [-1, 1])
    flat_labels = tf.reshape(y_tru, [-1, 1])
    _epsilon = tf.convert_to_tensor(EPSILON, tf.float32)

    if class_weight is not None:
        # transform to multi class
        flat_prd_0 = tf.constant(1., dtype=tf.float32) - flat_prd
        flat_multi_prd = tf.concat([flat_prd_0, flat_prd], axis=1)

        flat_labels_0 = tf.constant(1., dtype=tf.float32) - flat_labels
        flat_multi_label = tf.concat([flat_labels_0, flat_labels], axis=1)

        # transform back to logits
        flat_multi_prd = tf.clip_by_value(flat_multi_prd, _epsilon, 1 - _epsilon)
        flat_multi_prd = tf.log(flat_multi_prd / (1 - flat_multi_prd))

        # add class weighting
        class_weights = tf.constant(np.array(class_weight, np.float32))
        weighted_labels = tf.multiply(flat_multi_label, class_weights)
        # weight_map = tf.reduce_sum(weight_map, axis=1)
        weighted_loss = tf.nn.softmax_cross_entropy_with_logits(logits=flat_multi_prd, labels=weighted_labels)
        # weighted_loss = tf.multiply(loss_map, weight_map)
        loss = tf.reduce_mean(weighted_loss)

    else:
        # convert to logits
        flat_prd_logits = tf.clip_by_value(flat_prd, _epsilon, 1 - _epsilon)
        flat_prd_logits = tf.log(flat_prd_logits / (1 - flat_prd_logits))

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_prd_logits, labels=flat_labels))

    return loss
