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
from __future__ import division

from keras.models import Model
from keras import backend as K
from keras.layers.merge import add
from keras.layers import Input, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, Deconv2D, AveragePooling1D


def _bn_relu(input, with_bn):
    if with_bn:
        input = BatchNormalization()(input)
    return LeakyReLU()(input)


def _optimized_res_block_disc(input, dim):
    shortcut = AveragePooling1D(2)(input)
    shortcut = Conv1D(dim, kernel_size=1, padding="same")(shortcut)

    output = input
    output = Conv1D(dim, kernel_size=3, padding="same", kernel_initializer="he_normal")(output)
    output = LeakyReLU()(output)
    output = Conv1D(dim, kernel_size=3, padding="same", kernel_initializer="he_normal")(output)
    output = AveragePooling1D(2)(output)

    return add([shortcut, output])


def _residual_block(input, dim, kernel_size, with_bn, resample=None):
    output = input
    output = _bn_relu(output, with_bn)

    if resample == "up":
        input_shape = K.int_shape(input)
        output = Reshape(target_shape=input_shape[1:-1] + (1,) + input_shape[-1:])(output)
        output = Deconv2D(dim, strides=(2, 1), kernel_size=kernel_size,
                          padding="same", kernel_initializer="he_normal")(output)
        deconvolved_shape = K.int_shape(output)
        output = Reshape(deconvolved_shape[1:-2] + (deconvolved_shape[-1],))(output)
    else:
        output = Conv1D(dim, kernel_size=kernel_size, padding="same", kernel_initializer="he_normal")(output)

    output = _bn_relu(output, with_bn=with_bn)

    if resample == "down":
        output = Conv1D(dim, kernel_size=kernel_size, padding="same", kernel_initializer="he_normal")(output)
        output = AveragePooling1D(2)(output)
    else:
        output = Conv1D(dim, kernel_size=kernel_size, padding="same", kernel_initializer="he_normal")(output)

    input_shape = K.int_shape(input)
    output_shape = K.int_shape(output)

    if input_shape[-1] == output_shape[-1] and resample is None:
        shortcut = input
    else:
        if resample == "down":
            shortcut = Conv1D(output_shape[-1], kernel_size=1, padding="same")(input)
            shortcut = AveragePooling1D(2)(shortcut)
        elif resample == "up":
            input_shape = K.int_shape(input)
            shortcut = Reshape(input_shape[1:-1] + (1,) + input_shape[-1:])(input)
            shortcut = Deconv2D(dim, strides=(2, 1), kernel_size=kernel_size, padding="same",
                                kernel_initializer="he_normal")(shortcut)
            deconvolved_shape = K.int_shape(shortcut)
            shortcut = Reshape(deconvolved_shape[1:-2] + (deconvolved_shape[-1],))(shortcut)
        else:
            shortcut = Conv1D(output_shape[-1], kernel_size=1, padding="same")(input)

    return add([shortcut, output])


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, with_bn=True, num_layers=3, num_units=128):
        input = Input(shape=input_shape)

        output = _optimized_res_block_disc(input, num_units)  # Implicit "down"
        for _ in range(num_layers):
            num_units *= 2
            output = _residual_block(output, num_units, 3, with_bn, resample="down")
            output = _residual_block(output, num_units, 3, with_bn)

        output = _bn_relu(output, with_bn=with_bn)

        block_shape = K.int_shape(output)

        output = AveragePooling1D(pool_size=block_shape[-2], strides=1)(output)
        output = Flatten()(output)

        model = Model(inputs=input, outputs=output)
        # model.summary()
        return model
