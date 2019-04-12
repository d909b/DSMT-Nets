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
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.feature_extraction.settings import EfficientFCParameters
from util import make_tsfresh_transform_dataframe, collect_tsfresh_dataframe


def get_potential_auxiliary_tasks(x, y, n_jobs=4):
    x_df = collect_tsfresh_dataframe(x)
    x_transform_df = make_tsfresh_transform_dataframe(x)

    ts_features = RelevantFeatureAugmenter(column_id="id",
                                           column_value="value",
                                           n_jobs=n_jobs,
                                           filter_only_tsfresh_features=True,
                                           show_warnings=False,
                                           # We use EfficientFCParameters here for computational performance reasons.
                                           # ComprehensiveFCParameters may be used when performance is not critical.
                                           # ComprehensiveFCParameters was used in
                                           # "Distantly Supervised Multitask Learning in Critical Care".
                                           default_fc_parameters=EfficientFCParameters())

    ts_features.set_timeseries_container(x_df)
    ts_features.fit(X=x_transform_df, y=y)
    return ts_features
