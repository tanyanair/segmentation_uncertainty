import csv
from os.path import join
from os import makedirs

import matplotlib
import numpy as np
import pandas as pd
from scipy.misc import imsave

from bunet.utils.cca import cca_img_no_unc as cca_img
from bunet.utils.cca import remove_tiny_les
from bunet.utils.np_utils import sigmoid

matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ALPHA = 0.5
_MIN_SLICE = 5
_MAX_SLICE = 59


class CSVCallback:
    def __init__(self, csvfile, examples_per_epoch):
        self.__csvfile = csvfile
        self.__examples_per_epoch = examples_per_epoch
        self.__initialized = False

    def __call__(self, epoch, stats):
        if not self.__initialized:
            with open(self.__csvfile, 'w', newline='') as f:
                csvwriter = csv.writer(f, delimiter=',')
                csvwriter.writerow(
                    ['epoch', 'examples_seen', 'train_loss', 'train_dice', 'val_loss', 'val_dice', 'tpr', 'ppv'])
            self.__initialized = True
        with open(self.__csvfile, 'a', newline='') as f:
            csvwriter = csv.writer(f, delimiter=',')
            csvwriter.writerow([epoch,
                                epoch * self.__examples_per_epoch,
                                stats['train_loss'],
                                stats['train_dice'],
                                stats['loss'],
                                stats['dice'],
                                stats['tpr'],
                                stats['ppv']])


class PlotCallback:
    def __init__(self, csvfile, output_path):
        self.__csvfile = csvfile
        self.__out_dir = join(output_path, 'plots')
        makedirs(self.__out_dir, exist_ok=True)

    def __call__(self):
        data = np.asarray(pd.read_csv(self.__csvfile, header=None))
        nb_examples = np.asarray(data[1:, 1], np.int32)
        train_loss = np.asarray(data[1:, 2], np.float32)
        train_dice = np.asarray(data[1:, 3], np.float32)
        val_loss = np.asarray(data[1:, 4], np.float32)
        val_dice = np.asarray(data[1:, 5], np.float32)
        tpr = np.asarray(data[1:, 6], np.float32)
        ppv = np.asarray(data[1:, 7], np.float32)
        # fdr = np.asarray(data[1:,7], np.float32)
        # ppv = np.asarray(data[1:,8], np.float32)
        # fpr = np.asarray(data[1:,9], np.float32)
        # fnr = np.asarray(data[1:,10], np.float32)
        # f1_score = np.asarray(data[1:,11], np.float32)

        plt.plot(nb_examples, train_loss, '--b.', label='train_loss')
        plt.plot(nb_examples, val_loss, '--g.', label='val_loss')
        plt.ylabel('Loss')
        plt.xlabel('Training Examples Seen')
        plt.legend()
        plt.savefig(join(self.__out_dir, 'loss.png'))
        plt.close()

        plt.plot(nb_examples, train_dice, '--b.', label='train_dice')
        plt.plot(nb_examples, val_dice, '--g.', label='val_dice')
        plt.ylabel('DICE Coefficient')
        plt.xlabel('Training Examples Seen')
        plt.legend()
        plt.savefig(join(self.__out_dir, 'dice.png'))
        plt.close()

        plt.plot(nb_examples, tpr, '--b.', label='TPR')
        plt.plot(nb_examples, ppv, '--g.', label='PPV')
        plt.ylabel('TPR and PPV')
        plt.xlabel('Training Examples Seen')
        plt.legend()
        plt.savefig(join(self.__out_dir, 'tpr_ppv.png'))
        plt.close()


class ValidationLoss:
    def __init__(self, data_gen, nb_val_samples, cca_thresh, batch_size):
        """
        Callback that returns statistics from the validation data
        :param data_gen: generator that yields validation data
        :param nb_val_samples: number of validation samples to yield from data_gen
        :param cca_thresh: sigmoid threshold when performing connected component analysis
        :param batch_size: batch size of validation generator
        """
        self.__nb_val_samples = nb_val_samples
        self.__data_gen = data_gen
        self.__thresh = cca_thresh
        self.__batch_size = batch_size
        self.__nb_ex = nb_val_samples * batch_size

    def __call__(self, sess, net, epoch, cw):
        loss_sum = 0
        dice_sum = 0
        ppv_sum = 0
        tpr_sum = 0
        x_valid, y_valid = self.__data_gen.get_next()
        for i in range(self.__nb_val_samples):
            x_batch, y_batch = sess.run([x_valid, y_valid])
            mu, loss, dice = sess.run([net.predictor, net.loss, net.dice],
                                      feed_dict={net.x: x_batch, net.y: y_batch, net.keep_prob: .5,
                                                 net.class_weight: cw})
            for img in range(mu.shape[0]):
                stats = cca_img(mu[img, ..., 0], y_batch[img, ..., 0], self.__thresh)
                ppv_sum += 1 - stats['fdr']['all']
                tpr_sum += stats['tpr']['all']

            loss_sum += np.sum(loss)
            dice_sum += np.sum(dice)
        val_stats = {'loss': loss_sum / self.__nb_val_samples, 'dice': dice_sum / self.__nb_val_samples,
                     'ppv': ppv_sum / self.__nb_ex, 'tpr': tpr_sum / self.__nb_ex}
        return val_stats


class MIUncertaintyVisualizer:
    def __init__(self, sess, data_gen, out_dir, nb_mc=5, nb_imgs=12, dim=6, nb_ch=4):
        """
        MI uncertainty visualizer - use this callback when you only want to visualize one of the uncertainty measures
        :param sess: tensorflow session
        :param data_gen: generator of validation data
        :param out_dir: directory where output images will be stored
        :param nb_mc: number of Monte Carlo samples to take when calculating the uncertainty measure
        :param nb_imgs: number of images to visualize
        :param dim: dimension of output visualization callback
        :param nb_ch: number of channels in the input data
        """
        self._image_dir = join(out_dir, 'images')
        self._nb_imgs = nb_imgs
        makedirs(self._image_dir, exist_ok=True)
        self._nb_mc = nb_mc
        self._dim = dim
        self._data = None
        self._imgs = None
        self._imgs_tru = None
        self._sidx = None
        self._nb_ch = nb_ch
        self._prep_data(sess, data_gen)

    def __call__(self, sess, net, epoch):
        mus = []
        y_dummy = np.empty((1, 192, 192, 64, 1))
        for x in self._x_mc:
            mu = sess.run(net.predictor, feed_dict={net.x: x, net.y: y_dummy, net.keep_prob: .5, net.class_weight: 1})
            mus.append(mu)

        mus_mcs = np.asarray(mus)
        mus_mcs = mus_mcs.reshape((self._nb_mc, -1) + mus_mcs.shape[1:])
        y_prd = np.expand_dims(sigmoid(np.mean(mus_mcs, 0)), -1)  # nb_img, 192, 192, 64, 1
        mi = self._mi_uncertainty(sigmoid(mus_mcs))
        y_prd = y_prd[:, 0, ..., 0]  # 12, 1, 192, 192, 64, 1, 1
        canvas = self._get_canvas_mi_prd(y_prd, mi)

        # imsave(join(self._image_dir, "t1_{:03d}.png".format(epoch)), canvas[0])
        imsave(join(self._image_dir, "t2_{:03d}.png".format(epoch)), canvas[1])
        # imsave(join(self._image_dir, "flair_{:03d}.png".format(epoch)), canvas[2])

    @staticmethod
    def _mi_uncertainty(prd):
        # https://arxiv.org/pdf/1703.02910.pdf
        mcs = np.repeat(prd, 2, -1)
        mcs[..., 0] = 1 - mcs[..., 1]  # 10, nb_img, 192, 192, 64, 2

        entropy = -np.sum(np.mean(mcs, 0) * np.log(np.mean(mcs, 0) + 1e-5), -1)
        expected_entropy = -np.mean(np.sum(mcs * np.log(mcs + 1e-5), -1), 0)
        mi = entropy - expected_entropy
        return mi

    def _prep_data(self, sess, data_gen):
        print('preparing validation data... ')
        data = []
        xb = []
        yb = []
        i = 0
        x_valid, y_valid = data_gen.get_next()
        while len(yb) < self._nb_imgs:
            if i % 2 == 0:
                i += 1
                continue
            x, y = sess.run([x_valid, y_valid])
            ysum = y[0].sum()
            if ysum < 5:
                continue
            y = np.expand_dims(y[..., 0], -1)
            # remove lesions that are <=2 voxels (because they serve as an unfair comparison)
            y = remove_tiny_les(y)
            data.append((x, y))
            xb.append(x)
            yb.append(y)
            i += 1

        print('done preparing validation data')
        self._data = data

        x = np.asarray(xb)  # nb_im, bs, 192, 192, 64, 3
        y = np.asarray(yb)  # nb_im, bs, 192, 192, 64, 1

        self._x_mc = np.repeat(np.expand_dims(x[:, 0, ...], 0), self._nb_mc, 0). \
            reshape((x.shape[0] * self._nb_mc,) + x.shape[2:])  # nb_mc*nb_im, 192, 192, 64, 3
        self._x_mc = np.expand_dims(self._x_mc, 1)
        self._imgs_tru = 255 * np.repeat(np.expand_dims(_ALPHA * x[:, 0, ...], -1), 3, -1) + (1 - _ALPHA) * \
            np.repeat(np.expand_dims(np.repeat(y[:, 0, ...], self._nb_ch, -1), -1), 3, -1) * [0, 255, 0]
        self._imgs = x[:, 0, ...]  # 50, 192, 192, 64, 3
        self._set_sidx(y)

    def _set_sidx(self, y):
        y_imgs = y[:, 0, ..., 0]
        sidx = np.argmax(y_imgs.sum(1).sum(1), -1)
        sidx[sidx < _MIN_SLICE] = 30
        sidx[sidx > _MAX_SLICE] = 30
        self._sidx = sidx

    def _get_canvas_mi_prd(self, y_prd, mi):
        sx, sy, sz, ch = y_prd.shape[1:]  # (50, 192, 192, 64, 1)
        dim = self._dim
        imgs_prd = 255 * np.repeat(np.expand_dims(_ALPHA * self._imgs, -1), 3, -1) + \
                   (1 - _ALPHA) * np.repeat(np.expand_dims(np.repeat(y_prd, self._nb_ch, -1), -1), 3, -1) * [255, 0, 0]
        # 50, 192, 192, 64, 3, 3

        mi = np.repeat(np.expand_dims(np.repeat(
            np.expand_dims(mi, -1), self._nb_ch, -1), -1), 3, -1) * [255, 0, 0] # 50, 192, 192, 64, 3
        mi = mi[:, 0]
        out = np.zeros((3 * y_prd.shape[0], sx, sy, self._nb_ch, 3))  # 3*12, 192, 192, 3, 3
        for i in range(len(imgs_prd)):
            out[3 * i] = self._imgs_tru[i, :, :, self._sidx[i], :, :]
            out[3 * i + 1] = imgs_prd[i, :, :, self._sidx[i], :, :]
            out[3 * i + 2] = mi[i, :, :, self._sidx[i], :, :]

        canvas = np.zeros((3, dim * sx, dim * sy, 3), dtype='uint8')
        count = 0
        for j in range(dim):
            for k in range(dim):
                canvas[:, (j * sx):(j * sx + sx), (k * sy):(k * sy + sy), :] = \
                    np.flip(out[count].transpose(2, 1, 0, 3), 1)
                count += 1
        return canvas


class AllUncertaintyVisualizer(MIUncertaintyVisualizer):
    """
    Visualizer for 4 uncertainty measures
    """
    def __call__(self, sess, net, epoch):
        mus = []
        log_vars = []
        for x in self._x_mc:
            mu, log_var = sess.run([net.predictor, net.log_variance], feed_dict={net.x: x, net.keep_prob: .5})
            mus.append(mu)
            log_vars.append(log_var)
        mus_mcs = np.asarray(mus)
        mus_mcs = mus_mcs.reshape((self._nb_mc, -1) + mus_mcs.shape[1:])
        log_var_mcs = np.asarray(log_vars).reshape((self._nb_mc, -1) + mus_mcs.shape[1:])

        y_prd = np.expand_dims(sigmoid(np.mean(mus_mcs, 0)), -1)
        mi = self._mi_uncertainty(sigmoid(mus_mcs))  # (nb_img, 1, 192, 192, 64)
        ent = self._entropy(sigmoid(mus_mcs))
        prd_var = self._prd_variance(log_var_mcs)
        prd_unc = self._prd_uncertainty(mus_mcs)

        y_prd = y_prd[:, 0, ..., 0]  # 12, 1, 192, 192, 64, 1, 1
        mi_canvas = self._get_unc_canvas(y_prd, mi)
        ent_canvas = self._get_unc_canvas(y_prd, ent)
        prd_var_canvas = self._get_unc_canvas(y_prd, prd_var)
        prd_unc_canvas = self._get_unc_canvas(y_prd, prd_unc)

        # imsave(join(self._image_dir, "t1_{:03d}.png".format(epoch)), canvas[0])
        imsave(join(self._image_dir, "t2_mi_{:03d}.png".format(epoch)), mi_canvas[1])
        imsave(join(self._image_dir, "t2_ent_{:03d}.png".format(epoch)), ent_canvas[1])
        imsave(join(self._image_dir, "t2_prd_var_{:03d}.png".format(epoch)), prd_var_canvas[1])
        imsave(join(self._image_dir, "t2_prd_unc_{:03d}.png".format(epoch)), prd_unc_canvas[1])

    @staticmethod
    def _entropy(prd_mcs):
        mcs = np.repeat(prd_mcs, 2, -1)
        mcs[..., 0] = 1 - mcs[..., 1]  # 10, 50, 192, 192, 64, 2
        return -np.sum(np.mean(mcs, 0) * np.log(np.mean(mcs, 0) + 1e-5), -1)

    @staticmethod
    def _prd_variance(log_var_mcs):
        return np.mean(np.exp(np.clip(log_var_mcs, -7, 7)), 0)[
            0, ..., 0]  # (nb_mc, nb_img, 192, 192, 64, 1) --> (nb_img, 192, 192, 64, 1)

    @staticmethod
    def _prd_uncertainty(mu_mcs):
        # = Var(mu_mcs)
        return np.mean(np.square(mu_mcs[..., 0]), 0) - np.square(np.mean(mu_mcs[..., 0], 0))

    def _get_unc_canvas(self, y_prd, unc):
        sx, sy, sz, ch = y_prd.shape[1:]  # (50, 192, 192, 64, 1)
        dim = self._dim
        imgs_prd = 255 * np.repeat(np.expand_dims(_ALPHA * self._imgs, -1), 3, -1) + \
            (1 - _ALPHA) * np.repeat(np.expand_dims(np.repeat(y_prd, self._nb_ch, -1), -1), 3, -1) * [255, 0, 0]
        # 50, 192, 192, 64, 3, 3

        unc = np.repeat(np.expand_dims(np.repeat(
            np.expand_dims(unc, -1), self._nb_ch, -1), -1), 3, -1) * [255, 0, 0]  # 50, 192, 192, 64, 3
        unc = unc[:, 0]
        out = np.zeros((3 * y_prd.shape[0], sx, sy, self._nb_ch, 3))  # 3*12, 192, 192, 3, 3
        for i in range(len(imgs_prd)):
            out[3 * i] = self._imgs_tru[i, :, :, self._sidx[i], :, :]
            out[3 * i + 1] = imgs_prd[i, :, :, self._sidx[i], :, :]
            out[3 * i + 2] = unc[i, :, :, self._sidx[i], :, :]

        canvas = np.zeros((self._nb_ch, dim * sx, dim * sy, 3), dtype='uint8')
        count = 0
        for j in range(dim):
            for k in range(dim):
                canvas[:, (j * sx):(j * sx + sx), (k * sy):(k * sy + sy), :] = \
                    np.flip(out[count].transpose(2, 1, 0, 3), 1)
                count += 1
        return canvas
