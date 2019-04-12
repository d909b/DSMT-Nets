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
from os.path import join
from parameters import parse_parameters
from data_access.fog_dataset import Dataset
from model.model_builder import ModelBuilder
from model.model_trainer import ModelTrainer
from sklearn.preprocessing import StandardScaler
from model.model_evaluation import ModelEvaluation
from data_access.auxiliary_tasks import get_potential_auxiliary_tasks
from data_access.auxiliary_task_provider import AuxiliaryTaskProvider


class TrainApplication(object):
    def __init__(self):
        self.args = parse_parameters()
        self.initialise()

    def initialise(self):
        np.random.seed(self.args["seed"])

    @staticmethod
    def get_signal_names():
        return ["ankle_x", "ankle_y", "ankle_z", "leg_x", "leg_y", "leg_z", "trunk_x", "trunk_y", "trunk_z"]

    def get_signal_lengths(self):
        samples_per_segment = int(np.rint(self.args["samples_per_segment"]))
        return [samples_per_segment] * len(TrainApplication.get_signal_names())

    @staticmethod
    def get_generator(x, y, aux=None):
        assert len(x) == len(y)
        if aux is not None:
            assert len(x) == len(aux)
        num_steps = len(x)

        def generator():
            while 1:
                if aux is not None:
                    for idx in np.random.permutation(len(x)):
                        sample_x, sample_y, sample_aux = x[idx], y[idx], aux[idx]
                        yield map(lambda x: np.expand_dims(np.expand_dims(np.array(x), axis=-1), axis=0),
                                  sample_x.T.tolist()), (
                            np.expand_dims(sample_y, axis=-1),
                            np.expand_dims(sample_aux, axis=0)
                        )
                else:
                    for idx in np.random.permutation(len(x)):
                        sample_x, sample_y = x[idx], y[idx]
                        yield map(lambda x: np.expand_dims(np.expand_dims(np.array(x), axis=-1), axis=0),
                                  sample_x.T.tolist()), np.expand_dims(sample_y, axis=-1)

        return generator(), num_steps

    def run(self):
        num_epochs = int(np.rint(self.args["num_epochs"]))
        learning_rate = float(self.args["learning_rate"])
        output_directory = self.args["output_directory"]
        num_units = int(np.rint(self.args["num_units"]))
        n_jobs = int(np.rint(self.args["n_jobs"]))
        samples_per_segment = int(np.rint(self.args["samples_per_segment"]))

        (x_train, y_train), (x_val, y_val), (x_test, y_test) = Dataset.load_dataset(self.args["dataset"],
                                                                                    samples_per_segment=samples_per_segment)

        # Standardise dataset.
        collapsed = np.vstack(x_train)
        min_val, max_val = np.min(collapsed, axis=0), np.max(collapsed, axis=0)
        x_train = map(lambda xx: (xx - min_val) / (max_val - min_val), x_train)
        x_val = map(lambda xx: (xx - min_val) / (max_val - min_val), x_val)
        x_test = map(lambda xx: (xx - min_val) / (max_val - min_val), x_test)

        use_random_selection = True
        for num_labels in [12, 25]:
            x_train_labelled, x_train_unlabelled = x_train[:num_labels], x_train
            y_train_labelled, y_train_unlabelled = y_train[:num_labels], y_train

            potential_aux_tasks = get_potential_auxiliary_tasks(x_train_labelled, y_train_labelled, n_jobs)

            for num_auxiliary_tasks in [0, 5]:
                checkpoint_path = join(output_directory, "model.npy")
                if num_auxiliary_tasks != 0:
                    task_provider = AuxiliaryTaskProvider(
                        use_random_selection, potential_aux_tasks, num_auxiliary_tasks
                    )

                    aux_train_labelled = task_provider.calculate_auxiliary_tasks(x_train_labelled)
                    aux_train_unlabelled = task_provider.calculate_auxiliary_tasks(x_train_unlabelled)

                    # Standardise auxiliary tasks.
                    scaler = StandardScaler()
                    scaler.fit(np.concatenate([aux_train_labelled, aux_train_unlabelled], axis=0))
                    print("scale:", scaler.scale_)
                    aux_train_labelled = scaler.transform(aux_train_labelled)
                    aux_train_unlabelled = scaler.transform(aux_train_unlabelled)
                    aux_val = scaler.transform(task_provider.calculate_auxiliary_tasks(x_val))
                    aux_test = scaler.transform(task_provider.calculate_auxiliary_tasks(x_test))

                if num_auxiliary_tasks != 0:
                    model = ModelBuilder.build_semisupervised_dnn(
                        signal_names=TrainApplication.get_signal_names(),
                        signal_lengths=self.get_signal_lengths(),
                        num_units=num_units,
                        learning_rate=learning_rate,
                        num_auxiliary_tasks=num_auxiliary_tasks,
                        use_extra_auxiliary=True,
                        use_highway=True,
                        with_bn=False,
                        print_summary=False
                    )

                    labelled_train_set, labelled_train_steps =\
                        TrainApplication.get_generator(x_train_labelled, y_train_labelled, aux_train_labelled)
                    unlabelled_train_set, unlabelled_train_steps =\
                        TrainApplication.get_generator(x_train_unlabelled, y_train_unlabelled, aux_train_unlabelled)

                    val_set, val_steps = TrainApplication.get_generator(x_val, y_val, aux_val)
                    test_set, test_steps = TrainApplication.get_generator(x_test, y_test, aux_test)

                    ModelTrainer.train_dnn_w_auxiliary(
                        model, unlabelled_train_set, labelled_train_set, val_set,
                        num_epochs, labelled_train_steps, val_steps,
                        checkpoint_path=checkpoint_path
                    )
                else:
                    model = ModelBuilder.build_semisupervised_dnn(
                        signal_names=TrainApplication.get_signal_names(),
                        signal_lengths=self.get_signal_lengths(),
                        num_units=num_units,
                        learning_rate=learning_rate,
                        num_auxiliary_tasks=num_auxiliary_tasks,
                        use_extra_auxiliary=False,
                        return_intermediary=False,
                        use_highway=True,
                        with_bn=False,
                        print_summary=False
                    )

                    labelled_train_set, labelled_train_steps =\
                        TrainApplication.get_generator(x_train_labelled, y_train_labelled)

                    val_set, val_steps = TrainApplication.get_generator(x_val, y_val)
                    test_set, test_steps = TrainApplication.get_generator(x_test, y_test)

                    ModelTrainer.train_dnn(
                        model, labelled_train_set, val_set,
                        num_epochs, labelled_train_steps, val_steps,
                        checkpoint_path=checkpoint_path
                    )

                if isinstance(model, tuple):
                    model = model[0]
                else:
                    model = model

                print("Evaluation with", num_labels, "labels and", num_auxiliary_tasks, "aux tasks.", file=sys.stderr)
                model.set_weights(np.load(checkpoint_path))
                _, threshold = ModelEvaluation.evaluate_semisupervised_dnn(model, val_set, val_steps,
                                                                           with_auxiliary_tasks=num_auxiliary_tasks != 0,
                                                                           use_extra_auxiliary=num_auxiliary_tasks != 0)
                ModelEvaluation.evaluate_semisupervised_dnn(model, test_set, test_steps,
                                                            with_auxiliary_tasks=num_auxiliary_tasks != 0,
                                                            use_extra_auxiliary=num_auxiliary_tasks != 0,
                                                            threshold=threshold)


if __name__ == "__main__":
    app = TrainApplication()
    app.run()
