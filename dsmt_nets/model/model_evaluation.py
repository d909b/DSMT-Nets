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
from bisect import bisect_right
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, confusion_matrix, roc_curve, \
    precision_recall_curve, auc


class ModelEvaluation(object):
    @staticmethod
    def evaluate_semisupervised_dnn(model, generator, num_steps,
                                    with_auxiliary_tasks=False,
                                    use_extra_auxiliary=False,
                                    threshold=-1):
        batch_size = 0
        y_pred, y_true = [], []
        for _ in range(num_steps):
            generator_outputs = next(generator)
            if len(generator_outputs) == 3:
                batch_input, labels_batch, sample_weight = generator_outputs
            else:
                batch_input, labels_batch = generator_outputs

            model_outputs = model.predict(batch_input)

            if with_auxiliary_tasks:
                batch_size = len(labels_batch[0])
            else:
                batch_size = len(labels_batch)

            y_pred.append(model_outputs)
            y_true.append(labels_batch)

        if with_auxiliary_tasks:
            y_true = map(lambda x: x[0], y_true)

        if not use_extra_auxiliary and with_auxiliary_tasks:
            y_pred = map(lambda x: x[0], y_pred)

        y_pred = np.vstack(y_pred)

        if y_pred.shape[-1] == 1:
            y_pred, y_true = np.squeeze(y_pred), np.hstack(y_true)
        else:
            y_pred, y_true = np.vstack(y_pred)[:, :-1], np.vstack(y_true)[:, :-1]
            y_pred = y_pred[:, 0]
            y_true = y_true[:, 0]

        assert y_true.shape[-1] == y_pred.shape[-1]
        assert y_true.shape[-1] == num_steps * batch_size

        try:
            auc_score = roc_auc_score(y_true, y_pred)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)

            if threshold == -1.0:
                # Choose optimal threshold based on closest-to-top-left selection on ROC curve.
                optimal_threshold_idx = np.argmin(np.linalg.norm(np.stack((fpr, tpr)).T -
                                                                 np.repeat([[0., 1.]], fpr.shape[0], axis=0), axis=1))
                threshold = thresholds[optimal_threshold_idx]

            y_pred_thresholded = (y_pred > threshold).astype(np.int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresholded).ravel()

            sens_at_95spec_idx = bisect_right(fpr, 0.05)
            if sens_at_95spec_idx == 0:
                # Report 0.0 if specificity goal can not be met.
                sens_at_95spec = 0.0
            else:
                sens_at_95spec = tpr[sens_at_95spec_idx - 1]

            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            auprc_score = auc(recall, precision, reorder=False)

            print("INFO: Validated with AUC =", auc_score,
                  ", with AUPR =", auprc_score,
                  ", with accuracy =", accuracy_score(y_true, y_pred_thresholded),
                  ", with mean = ", np.mean(y_true),
                  ", with f1 =", f1_score(y_true, y_pred_thresholded),
                  ", with specificity =", float(tn) / (tn+fp),
                  ", with sensitivity = ", recall_score(y_true, y_pred_thresholded),
                  ", with sens@95spec = ", sens_at_95spec,
                  ", and n = ", len(y_true),
                  ", and threshold = ", threshold,
                  file=sys.stderr)
        except:
            print("WARN: Score calculation failed. Most likely, there was only one class present in y_true.",
                  file=sys.stderr)
            auc_score = 0

        return auc_score, threshold
