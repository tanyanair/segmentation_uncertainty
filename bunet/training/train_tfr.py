import os
import logging

import tensorflow as tf
from bunet.training.callbacks import CSVCallback, PlotCallback, ValidationLoss, AllUncertaintyVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
EPSILON = 1e-5


class Trainer(object):
    def __init__(self, net, opt_kwargs={}, wd=0., batch_size=2):
        """
        Trains a BUNet instance
        :param net: the bunet instance to train
        :param dict opt_kwargs: kwargs passed to optimizer (lr and decay)
        :param float wd: L2 weight decay parameter
        :param int batch_size: batch size
        """
        self.net = net
        self.opt_kwargs = opt_kwargs
        self.wd = wd
        self.batch_size = batch_size

    def _get_optimizer(self, training_iters, global_step):
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
        return optimizer

    def train(self, train_gen, val_gen, nb_val_steps, output_path, steps_per_epoch=10, epochs=100, dropout=None,
              restore_path="", write_graph=False, viz=True, cca_thresh=0.5, class_weight=None):
        """
        Launches training process
        """
        save_path = os.path.join(output_path, "model.cpkt")
        # initialize variables
        global_step = tf.Variable(0, trainable=False)
        self.optimizer = self._get_optimizer(steps_per_epoch, global_step)
        init_global = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True, force_gpu_compatible=False),
            allow_soft_placement=True)
        x_train, y_train = train_gen.get_next()

        with tf.Session(config=config) as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init_global)
            sess.run(init_local)
            sess.run(train_gen.initializer)
            sess.run(val_gen.initializer)

            if restore_path is not "":
                self.net.restore(sess, tf.train.latest_checkpoint(restore_path))

            loss_cbk = ValidationLoss(val_gen, nb_val_steps, cca_thresh, self.batch_size)
            csv_cbk = CSVCallback(os.path.join(output_path, 'validation.csv'), steps_per_epoch * self.batch_size)
            plot_cbk = PlotCallback(os.path.join(output_path, 'validation.csv'), output_path)
            if viz:
                viz_cbk = AllUncertaintyVisualizer(sess, val_gen, output_path)

            logging.info("Start optimization")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            cw = class_weight['start']
            min_loss = 1000
            for epoch in range(epochs):
                logging.info("Class Weighting {}".format(cw))
                total_loss = 0
                total_dice = 0
                for step in range((epoch * steps_per_epoch), ((epoch + 1) * steps_per_epoch)):
                    batch_x, batch_y = sess.run([x_train, y_train])
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
                cw *= class_weight['decay']
                if cw < class_weight['stop']:
                    cw = class_weight['stop']
                train_stats = {'train_loss': total_loss / steps_per_epoch,
                               'train_dice': total_dice / steps_per_epoch,
                               'lr': lr,
                               'prd_min': prd_min,
                               'prd_max': prd_max}
                self.output_epoch_stats(epoch + 1, train_stats)
                if viz:
                    viz_cbk(sess, self.net, epoch)
                val_stats = loss_cbk(sess, self.net, epoch + 1, cw)
                val_stats_str = "      Validation "
                for key, value in val_stats.items():
                    val_stats_str += "{}= {:.6f}    ".format(key, value)
                val_stats.update(train_stats)
                csv_cbk(epoch + 1, val_stats)
                plot_cbk()
                logging.info(val_stats_str)

                if min_loss > val_stats['loss']:
                    logging.info("      New best epoch! Saving model")
                    min_loss = val_stats['loss']
                    save_path = self.net.save(sess, save_path)

            coord.request_stop()
            coord.join(threads)
            logging.info("Optimization Finished!")

    @staticmethod
    def output_epoch_stats(epoch, train_stats):
        logging.info("Epoch {}, Average loss: {:.8f},   Average dice: {:.8f},   learning rate: {:.8f},  "
                     "prd_min: {:.6f}, prd_max: {:.6f}".
                     format(epoch, train_stats['train_loss'], train_stats['train_dice'], train_stats['lr'],
                            train_stats['prd_min'], train_stats['prd_max']))
