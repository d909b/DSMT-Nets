"""
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import csv
import numpy as np
from os import listdir
from itertools import groupby
from os.path import isfile, join
from util import ArrayIndexableList
from sklearn.model_selection import StratifiedShuffleSplit


class Dataset(object):
    @staticmethod
    def load_data_file(file_path):
        rows = []
        with open(file_path, "r") as data_file:
            reader = csv.reader(data_file, delimiter=" ")
            for row in reader:
                rows.append(row)
        return rows

    @staticmethod
    def get_target_column(x):
        return x[-1]

    @staticmethod
    def remove_timestamp_and_target_column(data_array):
        return data_array[:, 1:-1]

    @staticmethod
    def to_segments(rows):
        groups = []
        for key, group in groupby(rows, key=Dataset.get_target_column):
            key = int(key)
            if key == 0:
                continue  # Skip segments not part of experiment.
            else:
                key = 0 if key == 1 else 1  # 0 = control, 1 = case (freezing of gait)
            segment_data = Dataset.remove_timestamp_and_target_column(np.array(list(group), dtype=np.float32))
            groups.append((int(key), segment_data))
        return groups

    @staticmethod
    def load_segments(folder):
        segments = []
        for data_file in listdir(folder):
            data_file_path = join(folder, data_file)
            if isfile(data_file_path) and data_file.endswith(".txt"):
                rows = Dataset.load_data_file(data_file_path)
                segments += Dataset.to_segments(rows)
        return segments

    @staticmethod
    def split_segments(segments, validation_fraction, test_fraction, random_seed=909):
        segment_data = ArrayIndexableList(map(lambda x: x[1], segments))
        labels = np.array(map(lambda x: x[0], segments))

        non_fog_segments = np.where(labels == 0)[0]
        fog_segments = np.where(labels == 1)[0]

        # Balance dataset using ratio of 4:1 (original dataset is ~50:1.
        segment_data = ArrayIndexableList(segment_data[np.concatenate((fog_segments, non_fog_segments[:len(fog_segments)*4]))])
        labels = np.concatenate(([1]*len(fog_segments), [0]*(len(fog_segments)*4)))

        num_segments = len(segment_data)
        num_validation_segments = int(np.floor(num_segments * validation_fraction))
        num_test_segments = int(np.floor(num_segments * test_fraction))
        test_sss = StratifiedShuffleSplit(n_splits=1, test_size=num_test_segments, random_state=random_seed)
        rest_indices, test_indices = next(test_sss.split(segment_data, labels))

        rest_segment_data = ArrayIndexableList(segment_data[rest_indices])

        val_sss = StratifiedShuffleSplit(n_splits=1, test_size=num_validation_segments, random_state=random_seed)
        train_indices, val_indices = next(val_sss.split(rest_segment_data, labels[rest_indices]))

        return (rest_segment_data[train_indices], labels[rest_indices][train_indices]), \
               (rest_segment_data[val_indices], labels[rest_indices][val_indices]), \
               (segment_data[test_indices], labels[test_indices])

    @staticmethod
    def load_dataset(folder, validation_fraction=0.1, test_fraction=0.2, samples_per_segment=64):
        segments = Dataset.load_segments(folder)

        windows = []
        for label, segment_data in segments:
            num_windows_before = len(windows)
            num_windows = int(np.floor(len(segment_data) / samples_per_segment))
            for i in range(num_windows):
                windows.append((label, segment_data[i*samples_per_segment:(i+1)*samples_per_segment]))
            num_windows_after = len(windows)
            assert num_windows_after == num_windows_before + num_windows

        window_folds = Dataset.split_segments(windows, validation_fraction, test_fraction)
        for fold_data, fold_labels in window_folds:
            print("Fold", len(fold_data), " : ", np.mean(fold_labels))
        return window_folds
