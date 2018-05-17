import h5py
from os.path import join
import numpy as np

T2_LES = 4


class Reader(object):
    def __init__(self, file_name, patch_id, modalities):
        self._f = h5py.File(file_name, "r")
        self._modalities = modalities
        self._patch_id = patch_id

    def get_subjects(self):
        return list(self._f.keys())

    def get_time_points(self, subject):
        time_points = list(self._f[subject].keys())
        out = []
        for time_point in time_points:
            l = list(self._f[join(subject, time_point)].keys())
            if self._patch_id in l or self._patch_id == l:
                out.append(time_point)
        return out

    def get_data(self, subject, time_point):
        return self._f[join(subject, time_point, self._patch_id)][self._modalities, :, :, :]

    def get_label(self, subject, time_point):
        return np.expand_dims(self._f[join(subject, time_point, self._patch_id)][T2_LES, :, :, :], 0)

    def get_data_label_pair(self, subject, time_point):
        x = self._f[join(subject, time_point, self._patch_id)][:T2_LES, :, :, :]
        x = x[self._modalities, :, :, :]
        y = np.expand_dims(self._f[join(subject, time_point, self._patch_id)][T2_LES, :, :, :], 0)
        return x, y

    def get_score(self, subject, time_point):
        if 'edss' in self._f[join(subject, time_point)].attrs:
            return float(self._f[join(subject, time_point)].attrs["edss"])
        else:
            return float(-1)

    def get_data_score_pair(self, subject, time_point):
        return self.get_data(subject, time_point), self.get_score(subject, time_point)

    def get_patch_id(self):
        return self._patch_id
