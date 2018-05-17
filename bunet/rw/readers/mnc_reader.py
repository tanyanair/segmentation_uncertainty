import nibabel as nib
from os.path import join
import numpy as np

IMG_TAG = "_norm_ANAT-brain_ISPC-stx152lsq6.mnc.gz"
LES_TAG = "_ct2f_ISPC-stx152lsq6.mnc.gz"
GAD_TAG = "_gvf_ISPC-stx152lsq6.mnc.gz"
DATASET_TAG = "MS-LAQ-302-STX_"
TIME_POINTS = ['m0', 'm12', 'm24']
MODALITIES = ['t1p', 't2w', 'flr', 'pdw']

class Reader(object):
    def __init__(self, data_path, modalities=MODALITIES, time_points=TIME_POINTS):
        self._data_path = data_path
        self._modalities = modalities
        self._tps = time_points

        self._sites = [i for i in listdir(data_path)]
        subjs = [[site,subj] for site in sites for subj in listdir(join(d,site)) if '.scannerdb' not in subj]
        self._subj_list = []
        #print(len(subjs)) # 1401
        
        mod = {}
        [mod.update({i:{'min':1000, 'max':0}}) for i in ['flr', 'pdw', 't1', 't2']]
        count = 0
        start = timer()
        for site,subj in subjs:
            for tp in self._tps:
                if isdir(join(d,site,subj,tp)):
                    file_tag = "_norm_ANAT-brain_ISPC-stx152lsq6.mnc.gz"
                    ct2f = join(d,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+LES_TAG)
                    modality_path = {}
                    for m in self._modalities:
                        modality_path.update{m:join(d,site,subj,tp,'classifier_files', DATASET_TAG+site+"_"+subj+"_"+tp+"_"+m+IMG_TAG)}
                    data_exists = [True for key,value in modality_path.items() if exists(value) else False]
                    if np.all(data_exists):
                        count += 1
                        self._subj_list.append(subj)
                        self._subj_paths.append([site,subj])



    def get_subjects(self):
        return self._subj_list

    def get_time_points(self, subject):
        tps = []
        for subj in self._subj_paths:
            for tp in TIME_POINTS:
                if isdir(join(subj,tp)):
                    tps.append(tp)
        return tps

    def get_data(self, subject, time_point):
        site,subj = [i for i in self._subj_paths if subject in i[1]][0]
        data = []
        for m in MODALITIES:
            if m in self._modalities:
                data.append(nib.load(join(d,site,subj,time_point,'classifier_files', DATASET_TAG+site+"_"+subj+"_"+time_point+"_"+m+IMG_TAG)).get_data().astype(np.float32))
        return np.asarray(data,np.float32)

    def get_label(self, subject, time_point):
        site,subj = [i for i in self._subj_paths if subject in i[1]][0]
        data = nib.load(join(d,site,subj,time_point,DATASET_TAG+site+"_"+subj+"_"+time_point+"_"+LES_TAG)).get_data().astype(np.int32)
        return data

    def get_data_label_pair(self, subject, time_point):
        site,subj = [i for i in self._subj_paths if subject in i[1]][0]
        x = []
        for m in MODALITIES:
            if m in self._modalities:
                x.append(nib.load(join(d,site,subj,time_point,'classifier_files', DATASET_TAG+site+"_"+subj+"_"+time_point+"_"+m+IMG_TAG)).get_data().astype(np.float32))
        x = np.asarray(data)
        y = np.expand_dims(nib.load(join(d,site,subj,time_point,DATASET_TAG+site+"_"+subj+"_"+time_point+"_"+m+IMG_TAG)).get_data().astype(np.int32),0)

        return x, y
