from abc import abstractmethod
from random import shuffle

import numpy as np

from tf_unet.rw.readers.hdf5_reader import Reader as H5Reader


class BrainDataProvider:
    """
    Data provider for multimodal brain MRI.
    Usage:
    :param h5_path: path to hdf5 file
    :param config: dict to config options
    """
    def __init__(self, h5_path, config):
        self._r = H5Reader(h5_path, 'brains', tuple(config.get('modalities', [0, 1, 2])))
        self._mode = config.get('mode')
        self._shuffle = config.get('shuffle', True)
        self._augment = config.get('augment', False)
        self._subjects = self._r.get_subjects()
        self._nb_folds = config.get('nb-folds', 10)

        fold_length = len(self._subjects) // self._nb_folds
        if self._shuffle:
            shuffle(self._subjects)
        self._subjects = self._rotate(self._subjects, config.get('fold', 0) * fold_length)
        train_idx = (self._nb_folds - 2) * fold_length
        valid_idx = (self._nb_folds - 1) * fold_length
        if self._mode == 'train':
            self._subjects = self._subjects[:train_idx]
        elif self._mode == 'valid':
            self._subjects = self._subjects[train_idx:valid_idx]
        elif self._mode == 'test':
            self._subjects = self._subjects[valid_idx:]

    def __len__(self):
        return len(self._subjects)

    @staticmethod
    def _process_batch(x, dtype):
        x = np.asarray(x, dtype).transpose(0, 4, 3, 2, 1)  # (bs, z, y, x, modality)
        x = x[:, 12:204, 12:204, :, :]
        x = np.clip(x, 0., 1.)
        x = np.pad(x, ((0, 0), (0, 0), (0, 0), (2, 2), (0, 0)), 'constant', constant_values=0)
        return x
    
    @staticmethod
    def _rotate(l, n):
        return l[-n:] + l[:-n]
    
    @abstractmethod
    def get_generator(self, batch_size):
        raise NotImplementedError


class BrainVolumeDataProvider(BrainDataProvider):
    def get_generator(self, batch_size):
        x_batch = []
        y_batch = []
        count = 0
        while True:
            if self._shuffle:
                shuffle(self._subjects)
            for subject in self._subjects:
                tps = self._r.get_time_points(subject)
                if self._shuffle:
                    shuffle(tps)
                for tp in tps:
                    x_batch.append(self._r.get_data(subject, tp))
                    y_batch.append(self._r.get_label(subject, tp))
                    count += 1
                    if count == batch_size:
                        x = self._process_batch(x_batch, np.float32)  # (1, 192, 192, 64, 3)
                        y = self._process_batch(y_batch, np.float32)

                        yield x, y
                        x_batch = []
                        y_batch = []
                        count = 0

    def get_test_generator(self, batch_size):
        """
        Generator that yields examples (subj, time_point, x_batch, y_batch) for use during model evaluation
        :param batch_size: batch size of examples to yield
        """
        x_batch = []
        y_batch = []
        s_batch = []
        m_batch = []
        count = 0
        if self._shuffle:
            shuffle(self._subjects)
        for subject in self._subjects:
            tps = self._r.get_time_points(subject)
            if self._shuffle:
                shuffle(tps)
            for tp in tps:
                x_batch.append(self._r.get_data(subject, tp))
                y_batch.append(self._r.get_label(subject, tp))
                s_batch.append(subject)
                m_batch.append(tp)
                count += 1
                if count == batch_size:
                    x = self._process_batch(x_batch, np.float32)  # (1, 192, 192, 64, 3)
                    y = self._process_batch(y_batch, np.float32)
                    s = np.asarray(s_batch)
                    tp = np.asarray(m_batch)

                    yield s, tp, x, y
                    x_batch = []
                    y_batch = []
                    s_batch = []
                    m_batch = []
                    count = 0
