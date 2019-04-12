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
from __future__ import print_function

import sys
import numpy as np
from util import ArrayIndexableList, make_tsfresh_transform_dataframe, collect_tsfresh_dataframe


class AuxiliaryTaskProvider(object):
    def __init__(self, use_random_selection, auxiliary_feature_provider, num_tasks, random_seed=909):
        self.random_seed = random_seed
        self.use_random_selection = use_random_selection
        self.num_tasks = num_tasks
        self.auxiliary_feature_provider = auxiliary_feature_provider
        self.auxiliary_feature_provider = self.initialise_auxiliary_feature_provider()

    def get_selection(self):
        if self.use_random_selection:
            num_relevant = len(self.auxiliary_feature_provider.feature_selector.relevant_features)
            if num_relevant < self.num_tasks:
                print("WARN: Not enough relevant tasks for selection will consider irrelevant tasks to fill the quota.",
                      file=sys.stderr)
                num_relevant = len(self.auxiliary_feature_provider.feature_selector.features)

            np.random.seed(self.random_seed)  # Fix the seed to ensure deterministic split.
            return np.random.permutation(num_relevant).astype(int)[:self.num_tasks].tolist()
        else:
            return slice(0, self.num_tasks)

    def initialise_auxiliary_feature_provider(self):
        selection = self.get_selection()

        self.auxiliary_feature_provider.feature_selector.relevant_features = \
            ArrayIndexableList(self.auxiliary_feature_provider.feature_selector.features)
        self.auxiliary_feature_provider.feature_selector.relevant_features = \
            self.auxiliary_feature_provider.feature_selector.relevant_features[selection]

        assert len(self.auxiliary_feature_provider.feature_selector.relevant_features) == self.num_tasks

        return self.auxiliary_feature_provider

    def calculate_auxiliary_tasks(self, samples):
        test_df = collect_tsfresh_dataframe(samples)
        self.auxiliary_feature_provider.set_timeseries_container(test_df)

        transformed_data = self.auxiliary_feature_provider.transform(
            make_tsfresh_transform_dataframe(samples)
        )
        transformed_data = transformed_data.values

        assert transformed_data.shape[-1] >= self.num_tasks
        assert transformed_data.shape[0] == len(samples)

        return transformed_data[:, :self.num_tasks]
