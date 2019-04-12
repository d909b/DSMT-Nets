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

import os
from argparse import ArgumentParser, Action, ArgumentTypeError


class ReadableDir(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise ArgumentTypeError("readable_dir:{} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise ArgumentTypeError("readable_dir:{} is not a readable dir".format(prospective_dir))


def parse_parameters():
    parser = ArgumentParser(description='.')

    parser.add_argument("--dataset", action=ReadableDir, required=True,
                        help="Folder containing the data set to be loaded.")
    parser.add_argument("--output_directory", action=ReadableDir, required=True, default="./",
                        help="Folder to save output files to.")
    parser.add_argument("--seed", type=int, default=909,
                        help="Seed for the random number generator.")
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="Number of parallel compute threads to use in parallelised sections.")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="How many epochs to train for.")
    parser.add_argument("--num_units", type=int, default=256,
                        help="How many neurons to use per hidden layer in neural nets.")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="The learning rate to use for training neural nets.")
    parser.add_argument("--samples_per_segment", type=int, default=512,
                        help="How many samples to use per recording segment.")

    return vars(parser.parse_args())
