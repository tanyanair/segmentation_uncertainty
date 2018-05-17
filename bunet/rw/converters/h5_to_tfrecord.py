import h5py
import numpy as np
import tensorflow as tf
from os.path import join


class Converter:
    def __init__(self, h5_file, tfrecord_file, out_dir, img_dtype=np.float32, label_dtype=np.int16):
        self._f = h5py.File(h5_file, "r")
        self._tfrecord = tfrecord_file
        self._out_dir = out_dir
        self._img_dtype = img_dtype
        self._label_dtype = label_dtype
        self._data_shape = [60, 256, 256]

    @staticmethod
    def _bytes_feature(x):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))

    @staticmethod
    def _int64_feature(x):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))

    def run(self):
        shape = {0: self._int64_feature(self._data_shape[0]),
                 1: self._int64_feature(self._data_shape[1]),
                 2: self._int64_feature(self._data_shape[2])}

        writer = tf.python_io.TFRecordWriter(join(self._out_dir, self._tfrecord),
                                             options=tf.python_io.TFRecordOptions('ZLIB'))
        with self._f as f:
            for s in f.keys():
                for tp in f[s].keys():
                    img = np.asarray(f[s][tp]['brains'][:-1, ...], self._img_dtype)
                    label = np.asarray(f[s][tp]['brains'][-1, ...], self._label_dtype)

                    img_raw = img.tostring()
                    label_raw = label.tostring()

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'dim0': shape[0],
                        'dim1': shape[1],
                        'dim2': shape[2],
                        'img': self._bytes_feature(img_raw),
                        'label': self._bytes_feature(label_raw)}))

                    writer.write(example.SerializeToString())

