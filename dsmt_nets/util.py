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

import numpy as np
import pandas as pd


class ArrayIndexableList(list):
    def __getitem__(self, keys):
        if isinstance(keys, (int, slice)):
            return list.__getitem__(self, keys)
        return [self[k] for k in keys]


def get_list_in_equally_sized_chunks(list_object, num_chunks):
    for i in range(0, len(list_object), num_chunks):
        yield list_object[i:i + num_chunks]


# Transforms a dataset __x__ into the format required by the tsfresh library.
def collect_tsfresh_dataframe(x):
    x = map(lambda xx: xx.T, x)

    all_df = {}
    num_time_series = len(x[0])
    for i in range(num_time_series):
        full_df = pd.DataFrame(columns=["id", "time", "value"])
        for sample in range(len(x)):
            len_time_series = len(x[sample][i])
            id = [sample] * len_time_series
            timestamps = range(len_time_series)
            values = np.squeeze(x[sample][i])
            df = pd.DataFrame(data=np.array((id, timestamps, values)).T,
                              columns=["id", "time", "value"])
            full_df = full_df.append(df)
        all_df[str(i)] = full_df

    return all_df


def make_tsfresh_transform_dataframe(x_vals):
    return pd.DataFrame(data=np.expand_dims(range(len(x_vals)), axis=-1), columns=["id"])
