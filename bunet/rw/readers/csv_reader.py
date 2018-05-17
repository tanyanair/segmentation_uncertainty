import pandas as pd
import numpy as np


class Reader:
    def __init__(self, file_name):
        self._f = np.asarray(pd.read_csv(file_name, header=None))
        self._stats = None
        self._build_stats()

    def _build_stats(self):
        stats = {}
        total_large = 0

        for i, line in enumerate(self._f):
            if i == 0:
                continue
            # site = line[0]
            subj = line[1]
            drug = line[2]
            month = line[3]
            nb_les = line[4]
            tlv = line[5]
            # mean_lv = line[6]
            # med_lv = line[7]
            # std_lv = line[8]
            nb_tiny = line[10]
            nb_small = line[11]
            nb_med = line[12]
            nb_large = line[13]
            # activity = line[-1]
            if subj not in stats.keys():
                stats[subj] = {}

            stats[subj][month] = {}
            stats[subj][month]['drug'] = drug
            stats[subj][month]['nb_les'] = int(nb_les)
            stats[subj][month]['tlv'] = float(tlv)
            stats[subj][month]['nb_tiny'] = int(nb_tiny)
            stats[subj][month]['nb_small'] = int(nb_small)
            stats[subj][month]['nb_med'] = int(nb_med)
            stats[subj][month]['nb_large'] = int(nb_large)
            if int(nb_large) >= 1:
                total_large += 1

        self._stats = stats

    def get_time_point_stat(self, subject, time_point, stat):
        if subject not in self._stats.keys():
            Warning('subject {} not in subject list'.format(subject))
            return 0
        elif time_point not in self._stats[subject].keys():
            Warning('time point {} not in time point list for subject {}\n'
                    '    available time points: {}'.format(time_point, subject, self._stats[subject].keys()))
            return 0
        elif stat not in self._stats[subject][time_point].keys():
            Warning('statistic {} not available for subject {} time point {}\n'
                    '    available statistics: {}'.format(stat, subject, time_point,
                                                          self._stats[subject][time_point].keys()))
            return 0
        return self._stats[subject][time_point][stat]
