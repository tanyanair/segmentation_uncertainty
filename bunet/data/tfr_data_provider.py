import numpy as np
import tensorflow as tf

_SEED = 5
_BUFFER_SIZE = 1350


class BrainDataProvider:
    def __init__(self, tfrecord_path, config):
        """
        Data provider for multimodal brain MRI.
        Usage:
        :param tfrecord_path: path to tfrecord
        :param config: dict to config options
        """
        self._tfrecord = tfrecord_path
        self._mode = config.get('mode')
        self._shuffle = config.get('shuffle', True)
        self._augment = config.get('augment', False)
        self._subjects = np.load(config.get('subj_list', '/path/to/subjects/list_in_npy_format.npy'))

        # get the unique integer subject identifier from the subject string
        self._subjects = [i[4:] for i in self._subjects]

        nb_train = round(len(self._subjects) * 0.8)
        nb_valid = round(len(self._subjects) * 0.1)
        if self._mode == 'train':
            self._subjects = self._subjects[:nb_train]
        elif self._mode == 'valid':
            self._subjects = self._subjects[nb_train:nb_train+nb_valid]
        elif self._mode == 'test':
            self._subjects = self._subjects[nb_train+nb_valid:]
        self._subjects = tf.constant(np.asarray(self._subjects, np.int32))


    @staticmethod
    def _parse_tfrecord(tfrecord):
        features = tf.parse_single_example(tfrecord,
                                           features={'img': tf.FixedLenFeature([], tf.string),
                                                     'label': tf.FixedLenFeature([], tf.string),
                                                     'subj': tf.FixedLenFeature([], tf.int64),
                                                     'time_point': tf.FixedLenFeature([], tf.int64)})
        subj = tf.cast(features['subj'], tf.int32)
        return subj, features['time_point'], features['img'], features['label']

    @staticmethod
    def _process_data(x):
        x = tf.transpose(x, [3, 2, 1, 0])  # (bs, z, y, x, modality)
        x = x[12:204, 12:204, :, :] # crop out the background voxels
        x = tf.pad(x, paddings=tf.constant([[0, 0], [0, 0], [2, 2], [0, 0]]),
                   mode='CONSTANT', constant_values=0)
        return x

    def _subj_filter(self, subj, tp, img, label):
        return tf.reduce_any(tf.equal(subj, self._subjects))

    def _remove_subj_tp(self, subj, tp, img, label):
        img = tf.decode_raw(img, tf.float32)
        label = tf.cast(tf.decode_raw(label, tf.int16), tf.float32)
        # hard coded shape because the images are a fixed size
        img = tf.reshape(img, tf.stack([4, 60, 256, 256]))
        label = tf.reshape(label, tf.stack([1, 60, 256, 256]))
        img = self._process_data(img)
        label = self._process_data(label)
        return img, label

    @abstractmethod
    def get_generator(self, batch_size, nb_epochs):
        raise NotImplementedError


class BrainVolumeDataProvider(BrainDataProvider):
    def get_generator(self, batch_size, nb_epochs):
        """
        Generator that yields examples (subj, time_point, x_batch, y_batch) for use during model training
        :param batch_size: batch size of examples to yield
        :param nb_epochs: number of epochs to generate
        """
        dataset = tf.data.TFRecordDataset([self._tfrecord])
        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=10)
        dataset = dataset.filter(self._subj_filter)
        dataset = dataset.map(self._remove_subj_tp, num_parallel_calls=10)
        if self._shuffle:
            dataset = dataset.shuffle(buffer_size=_BUFFER_SIZE, seed=_SEED)
        dataset = dataset.repeat(nb_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(50)
        iterator = dataset.make_initializable_iterator()
        return iterator

    def get_test_generator(self, batch_size=1, nb_epochs=1):
        """
        Generator that yields examples (subj, time_point, x_batch, y_batch) for use during model evaluation
        :param batch_size: batch size of examples to yield
        :param nb_epochs: number of epochs to generate
        """
        dataset = tf.data.TFRecordDataset(self._tfrecord)
        dataset = dataset.map(self._parse_tfrecord).filter(self._subj_filter)
        dataset = dataset.repeat(nb_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        return iterator
