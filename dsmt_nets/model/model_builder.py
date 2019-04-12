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

import keras.backend as K
import keras.initializers
from resnet import ResnetBuilder
from keras.optimizers import RMSprop
from keras.models import Model, Input
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation, Dropout, concatenate, Multiply, Add, Lambda, GaussianNoise, Reshape


class ModelBuilder(object):
    @staticmethod
    def build_mlp(model_dimension, input_dim=2, num_layers=2, input_name=None,
                  p_dropout=0.25, input_layer=None, with_bn=True):
        if input_layer is None:
            input_shape = input_dim,
            input_layer = Input(shape=input_shape, name=input_name)

        last_layer = input_layer
        for _ in range(num_layers):
            last_layer = Dense(model_dimension)(last_layer)
            if with_bn:
                last_layer = BatchNormalization()(last_layer)
            last_layer = Activation("relu")(last_layer)
            last_layer = Dropout(p_dropout)(last_layer)

        return input_layer, last_layer

    @staticmethod
    def build_highway_layers(input_layer, activation="relu", gate_bias=-3, num_layers=1, p_dropout=0.0, with_bn=True):
        dim = K.int_shape(input_layer)[-1]
        gate_bias_initializer = keras.initializers.Constant(gate_bias)
        for i in range(num_layers):
            gate = Dense(units=dim,
                         bias_initializer=gate_bias_initializer)(input_layer)
            gate = Activation("sigmoid")(gate)
            negated_gate = Lambda(
                lambda x: 1.0 - x,
                output_shape=(dim,))(gate)
            transformed = Dense(units=dim)(input_layer)
            if with_bn:
                transformed = BatchNormalization()(transformed)
            transformed = Activation(activation)(transformed)
            transformed = Dropout(p_dropout)(transformed)
            transformed_gated = Multiply()([gate, transformed])
            identity_gated = Multiply()([negated_gate, input_layer])
            input_layer = Add()([transformed_gated, identity_gated])
        return input_layer

    @staticmethod
    def get_optimiser(learning_rate):
        return RMSprop(learning_rate)

    @staticmethod
    def build_signal_expert(signal_name, signal_length, num_units, with_bn):
        resnet_model = ResnetBuilder.build(input_shape=(signal_length, 1), num_units=num_units, with_bn=with_bn)
        return resnet_model
        input_before = Input(shape=(signal_length, 1))
        last_layer = Reshape((-1,))(input_before)
        input_layer, output_layer = ModelBuilder.build_mlp(num_units, 64, input_layer=last_layer, with_bn=with_bn)
        # resnet_model.name = signal_name + "_resnet"
        return Model(input_before, output_layer)

    @staticmethod
    def build_perception_blocks(signal_names, signal_lengths, num_units, print_summary, with_bn=False):
        all_inputs, all_experts, data_streams = [], [], []
        for signal_name, signal_length in zip(signal_names, signal_lengths):
            signal_input = Input(shape=(signal_length, 1), name=signal_name + "_signal_input")

            expert_model = ModelBuilder.build_signal_expert(signal_name, signal_length, num_units, with_bn)

            if print_summary:
                expert_model.summary()

            all_inputs.append(signal_input)
            all_experts.append(expert_model)
            data_streams.append(expert_model(signal_input))
        return all_inputs, all_experts, data_streams

    @staticmethod
    def build_semisupervised_dnn(signal_names, signal_lengths, num_units, learning_rate, dropout=0.,
                                 use_highway=False, use_horizontal=True, num_layers=1, auxiliary_task_weight=1.0,
                                 use_extra_auxiliary=False, return_intermediary=True, use_linear_output=False,
                                 use_auxiliary_outputs=True, aux_noise_level=0.35,
                                 num_outputs=1, print_summary=False, num_auxiliary_tasks=0, with_bn=True):
        all_inputs, all_experts, data_streams = ModelBuilder.build_perception_blocks(signal_names,
                                                                                     signal_lengths,
                                                                                     num_units,
                                                                                     print_summary,
                                                                                     with_bn)

        combined_streams = concatenate(data_streams, axis=-1)

        auxiliary_tasks, auxiliary_losses = [], []
        if use_highway:
            last_layer = combined_streams

            highway_auxiliary_tasks, auxiliary_last_layers = [], []
            for i in range(num_auxiliary_tasks):
                if use_horizontal:
                    last_layer = combined_streams

                last_layer = Dense(num_units)(last_layer)
                last_layer = BatchNormalization()(last_layer)
                last_layer = Activation("relu")(last_layer)
                last_layer = Dropout(dropout)(last_layer)

                last_layer = ModelBuilder.build_highway_layers(input_layer=last_layer,
                                                               num_layers=num_layers,
                                                               p_dropout=dropout,
                                                               with_bn=with_bn)

                if use_auxiliary_outputs:
                    auxiliary_task = GaussianNoise(aux_noise_level)(last_layer)
                    auxiliary_task = Dense(1, name="highway_auxiliary_task_" + str(i),
                                           activation="linear")(auxiliary_task)
                    highway_auxiliary_tasks.append(auxiliary_task)

                auxiliary_last_layers.append(last_layer)

            if use_horizontal and num_auxiliary_tasks != 0:
                # combinator_mode = "serial"
                last_layer = concatenate([combined_streams] + auxiliary_last_layers)

                last_layer = Dense(num_units * 2)(last_layer)
                last_layer = BatchNormalization()(last_layer)
                last_layer = Activation("relu")(last_layer)
                last_layer = Dropout(dropout)(last_layer)

            last_layer = ModelBuilder.build_highway_layers(input_layer=last_layer,
                                                           num_layers=num_layers,
                                                           p_dropout=dropout,
                                                           with_bn=with_bn)

            if num_auxiliary_tasks != 0 and use_auxiliary_outputs:
                if num_auxiliary_tasks == 1:
                    combined_highway_auxiliary_tasks = highway_auxiliary_tasks[0]
                else:
                    combined_highway_auxiliary_tasks = concatenate(highway_auxiliary_tasks,
                                                                   axis=-1,
                                                                   name="aux_tasks_" + str(num_auxiliary_tasks))

                auxiliary_tasks.append(combined_highway_auxiliary_tasks)
                auxiliary_losses.append("mse")
        else:
            _, last_layer = ModelBuilder.build_mlp(model_dimension=num_units * len(signal_names),
                                                   input_layer=combined_streams,
                                                   num_layers=num_layers,
                                                   p_dropout=dropout,
                                                   with_bn=with_bn)

            if num_auxiliary_tasks != 0 and use_auxiliary_outputs:
                aux_task = Dense(num_auxiliary_tasks,
                                 name="aux_tasks_" + str(num_auxiliary_tasks),
                                 activation="linear")(last_layer)
                auxiliary_tasks.append(aux_task)
                auxiliary_losses.append("mse")

        intermediary_layer = last_layer

        main_output = Dense(num_outputs,
                            name="semisupervised_output",
                            activation="linear" if use_linear_output else "sigmoid")(last_layer)

        if len(auxiliary_losses) != 0:
            loss_weights = [1.0, auxiliary_task_weight]
        else:
            loss_weights = [1.0]

        if use_extra_auxiliary:
            semisupervised_dnn_model = Model(inputs=all_inputs, outputs=[main_output])
            semisupervised_dnn_model.compile(optimizer=ModelBuilder.get_optimiser(learning_rate),
                                             loss=["binary_crossentropy"],
                                             metrics=["accuracy"],
                                             loss_weights=[loss_weights[0]])

            auxiliary_model = Model(inputs=all_inputs, outputs=auxiliary_tasks)

            auxiliary_model.compile(optimizer=ModelBuilder.get_optimiser(learning_rate),
                                    loss=auxiliary_losses,
                                    loss_weights=[loss_weights[1]])

            return semisupervised_dnn_model, auxiliary_model
        else:
            semisupervised_dnn_model = Model(inputs=all_inputs, outputs=[main_output] + auxiliary_tasks)
            semisupervised_dnn_model.compile(optimizer=ModelBuilder.get_optimiser(learning_rate),
                                             loss=["binary_crossentropy"] + auxiliary_losses,
                                             metrics={"semisupervised_output": "accuracy"},
                                             loss_weights=loss_weights)

            if print_summary:
                semisupervised_dnn_model.summary()

            if return_intermediary:
                intermediary_dnn_model = Model(inputs=all_inputs, outputs=intermediary_layer)
                return semisupervised_dnn_model, intermediary_dnn_model
            else:
                return semisupervised_dnn_model

    @staticmethod
    def get_inputs(model, prefix):
        all_inputs = []
        for input_name, input_shape in zip(model.input_names, model.input_shape):
            all_inputs.append(Input(batch_shape=input_shape, name=prefix + input_name))
        return all_inputs
