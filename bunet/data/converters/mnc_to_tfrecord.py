import numpy as np
import tensorflow as tf
from os.path import join, exists, isdir
from os import listdir
from timeit import default_timer as timer
import nibabel as nib

DEFAULT_DATA_DIR = '/cim/data_raw/MS-LAQ-302-STX/imaging_data'
IMG_TAG = "_norm_ANAT-brain_ISPC-stx152lsq6.mnc.gz"
IMG_NII_TAG = "_norm_ANAT-brain_ISPC-stx152lsq6.nii"
LES_TAG = "_ct2f_ISPC-stx152lsq6.mnc.gz"
LES_NII_TAG = "_ct2f_ISPC-stx152lsq6.nii"
GAD_TAG = "_gvf_ISPC-stx152lsq6.mnc.gz"
DATASET_TAG = "MS-LAQ-302-STX_"
DEFAULT_TPS = ['m0', 'm12', 'm24']
MODALITIES = ['t1p', 't2w', 'flr', 'pdw']

NORM_CONSTANTS = {'t1p':{'max': 1020.0050537880522},'t2w':{'max': 1016.9968841077286},
                  'flr':{'max': 1014.9998077363241},'pdw':{'max': 1016.9996490424962}}

class Converter:
    def __init__(self, tfrecord, data_dir=DEFAULT_DATA_DIR, img_dtype=np.float32, label_dtype=np.int16):
        self._tfrecord = tfrecord
        self._data_dir = data_dir
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

        writer = tf.python_io.TFRecordWriter(self._tfrecord)#,
                                             #options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))

        sites = [i for i in listdir(self._data_dir)]
        subjs = [[site,subj] for site in sites for subj in listdir(join(self._data_dir,site)) if '.scannerdb' not in subj]
        #print(len(subjs)) # 1401
        count = 0
        start = timer()
        for site,subj in subjs:
            for tp in DEFAULT_TPS:
                if isdir(join(self._data_dir,site,subj,tp)):
                    ct2f_pth = join(self._data_dir,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+LES_TAG)
                    flr_pth = join(self._data_dir,site,subj,tp,'classifier_files', DATASET_TAG+site+"_"+subj+"_"+tp+"_flr"+IMG_TAG)
                    pdw_pth = join(self._data_dir,site,subj,tp,'classifier_files', DATASET_TAG+site+"_"+subj+"_"+tp+"_pdw"+IMG_TAG)
                    t1c_pth = join(self._data_dir,site,subj,tp,'classifier_files', DATASET_TAG+site+"_"+subj+"_"+tp+"_t1p"+IMG_TAG)
                    t2w_pth = join(self._data_dir,site,subj,tp,'classifier_files', DATASET_TAG+site+"_"+subj+"_"+tp+"_t2w"+IMG_TAG)
                    if np.all([exists(ct2f_pth), exists(flr_pth), exists(pdw_pth), exists(t1c_pth), exists(t2w_pth)]):
                        data = []
                        if subj == 'nii_9590710':
                            for m in MODALITIES:
                                img = nib.load(join(self._data_dir,site,subj,tp,'classifier_files', DATASET_TAG+site+"_"+subj+"_"+tp+"_"+m+IMG_NII_TAG)).get_data().astype(np.float32)
                                img = (img-img.min()) / (NORM_CONSTANTS[m]['max']-img.min())
                                img = np.clip(img,0.0,1.0)
                                data.append(img)
                            data = np.asarray(data, self._img_dtype).transpose(0,3,2,1)
                            label = np.expand_dims(nib.load(join(self._data_dir,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+LES_NII_TAG)).get_data().astype(self._label_dtype),0)
                            label = label.transpose(0,3,2,1)

                        else:
                            for m in MODALITIES:
                                img = nib.load(join(self._data_dir,site,subj,tp,'classifier_files', DATASET_TAG+site+"_"+subj+"_"+tp+"_"+m+IMG_TAG)).get_data().astype(np.float32)
                                img = (img-img.min()) / (NORM_CONSTANTS[m]['max']-img.min())
                                img = np.clip(img,0.0,1.0)
                                data.append(img)
                            data = np.asarray(data, self._img_dtype)
                            label = np.expand_dims(nib.load(join(self._data_dir,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+LES_TAG)).get_data().astype(self._label_dtype),0)

                        assert data.shape == (4,60,256,256), print("{} is not the right data shape".format(data.shape))
                        assert label.shape == (1,60,256,256), print("{} is not the right data shape".format(label.shape))
                        if (count+1) % 100 ==  0:
                            print('{} images complete {:.2f}m'.format(count+1,(timer()-start)/60) )
                        count += 1

                        data_raw = data.tostring()
                        label_raw = label.tostring()

                        example = tf.train.Example(features=tf.train.Features(feature={
                            'dim0': shape[0],
                            'dim1': shape[1],
                            'dim2': shape[2],
                            'subj': self._int64_feature(int(subj[4:])),
                            'time_point': self._int64_feature(int(tp[1:])),
                            'img': self._bytes_feature(data_raw),
                            'label': self._bytes_feature(label_raw)}))

                        writer.write(example.SerializeToString())
        print("wrote {} images to {}".format(count,self._tfrecord))
        writer.close()
        print('closed writer')
        sys.stdout.flush()
        print('cleared stdout')

if __name__=="__main__":
    c = Converter('/cim/tnair/mslaq5.tfrecord')
    c.run()
    print('Done')
