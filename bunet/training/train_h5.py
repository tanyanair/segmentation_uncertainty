import os
import shutil
import numpy as np
import logging

import tensorflow as tf
from tf_unet.callbacks import ValidationLoss
from tf_unet.callbacks import MIUncertaintyVisualizer as UncertaintyVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
EPSILON = 1e-5


class Trainer(object):
    """
    Trains a unet instance
    :param net: the bunet instance to train
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """

    def __init__(self, net, norm_grads=False, optimizer="momentum", opt_kwargs={}, wd=0.):
        self.net = net
        self.norm_grads = norm_grads
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.wd = wd

    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("lr", 0.2)
            decay_rate = self.opt_kwargs.pop("decay", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=training_iters,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.loss,
                                                                               global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("lr", 0.001)
            decay_rate = self.opt_kwargs.pop("decay", 0.95)
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=training_iters,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                               **self.opt_kwargs).minimize(self.net.loss,
                                                                           global_step=global_step)
        else:
            raise ValueError('optimizer must be `adam` or `momentum`, you passed: `{}`'.format(self.optimizer))

        return optimizer

    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0, trainable=False)
        self.optimizer = self._get_optimizer(training_iters, global_step)
        init = tf.global_variables_initializer()

        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, force_gpu_compatible=False),
            allow_soft_placement=True)

        return init, config

    def train(self, train_gen, val_gen, nb_val_steps, output_path, steps_per_epoch=10, epochs=100, dropout=0.75,
              display_step=1, restore=False, write_graph=False, prediction_path='prediction', viz=True):
        """
        Launches training process
        """
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path

        init, config = self._initialize(steps_per_epoch, output_path, restore, prediction_path)
        with tf.Session(config=config) as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            loss_cbk = ValidationLoss(val_gen, nb_val_steps)
            if viz:
                viz_cbk = UncertaintyVisualizer(val_gen, output_path)

            logging.info("Start optimization")

            cw = 800
            for epoch in range(epochs):
                logging.info("Class Weighting {}".format(cw))
                total_loss = 0
                total_dice = 0
                minibatch_loss = 0
                minibatch_dice = 0
                for step in range((epoch * steps_per_epoch), ((epoch + 1) * steps_per_epoch)):
                    batch_x, batch_y = next(train_gen)
                    logging.debug(
                        'batch_x.shape {}  batch_y.shape {}  dropout {}'.format(batch_x.shape, batch_y.shape, dropout))

                    # Run optimization
                    _, loss, dce, prd_min, prd_max, lr, gradients = sess.run((self.optimizer,
                                                                              self.net.loss,
                                                                              self.net.dice,
                                                                              self.net.prd_min,
                                                                              self.net.prd_max,
                                                                              self.learning_rate_node,
                                                                              self.net.gradients_node),
                                                                             feed_dict={self.net.x: batch_x,
                                                                                        self.net.y: batch_y,
                                                                                        self.net.keep_prob: dropout,
                                                                                        self.net.class_weight: cw})

                    total_loss += loss
                    total_dice += dce
                    minibatch_loss += loss
                    minibatch_dice += dce

                    if (step + 1) % display_step == 0:
                        self.output_minibatch_stats(step + 1, minibatch_loss / (step + 1), minibatch_dice / (step + 1))
                cw *= .8
                if cw < 4:
                    cw = 4
                self.output_epoch_stats(epoch + 1, total_loss, total_dice, steps_per_epoch, lr, prd_min, prd_max)
                if viz:
                    viz_cbk(sess, self.net, epoch)
                val_loss, val_dice = loss_cbk(sess, self.net, epoch + 1)
                logging.info("      Validation loss= {:.4f}   dice= {:.6f}".format(val_loss, val_dice))

                save_path = self.net.save(sess, save_path)
            logging.info("Optimization Finished!")

            return save_path

    @staticmethod
    def output_epoch_stats(epoch, total_loss, total_dice, training_iters, lr, prd_min, prd_max):
        # logging.info("Epoch {:}, Average loss: {:.4f},   Average dice: {:.6f},   learning rate: {:.6f}".
        logging.info("Epoch {}, Average loss: {:.8f},   Average dice: {:.8f},   learning rate: {:.8f},  "
                     "prd_min: {:.6f}, prd_max: {:.6f}".
                     format(epoch, (total_loss / training_iters), (total_dice / training_iters), lr, prd_min, prd_max))

    @staticmethod
    def output_minibatch_stats(step, loss, dce):
        # Calculate batch loss and dice
        # logging.info("Iter {:}, Minibatch Loss= {:.6f}, Minibatch Dice= {:.6f}".format(step, loss, -dce))
        logging.info("Iter {}, Minibatch Loss= {:.8f}, Minibatch Dice= {:.8f}".format(step, loss, dce))


def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))

    return avg_gradients


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0] * predictions.shape[1] * predictions.shape[2]))
