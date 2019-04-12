## DSMT-Nets: Distantly Supervised Multitask Networks

![Distantly Supervised Multitask Networks](http://schwabpatrick.com/img/dsmtn-overview.png)

Distantly supervised multitask networks (DSMT-Nets) are an approach to semi-supervised learning that utilises distant supervision through many auxiliary tasks.
This approach can be an effective approach to semi-supervised learning in domains where large numbers of diverse features are readily available for use as auxiliary tasks, e.g. in time series.
[Our paper](http://proceedings.mlr.press/v80/schwab18a.html) contains a detailed description of DSMT-Nets, and how they can be applied to a real-world problem with multivariate, high-resolution time series in the intensive care unit (ICU).

This repository contains a simplified educational example of how to use DSMT-Nets with the auxiliary tasks automatically selected using the [tsfresh](https://github.com/blue-yonder/tsfresh) library and a public dataset. The ICU dataset used in our paper is currently unfortunately not openly available due to its sensitive nature.

Contact: Patrick Schwab, ETH Zurich <patrick.schwab@hest.ethz.ch>

License: MIT; See LICENSE.txt

### Citation

If you reference or use our methodology, code or results in your work, please consider citing:

    @InProceedings{schwab18nottocrywolf,
      title = {{Not to Cry Wolf: Distantly Supervised Multitask Learning in Critical Care}},
      author = {Schwab, Patrick and Keller, Emanuela and Muroi, Carl and Mack, David J. and Str{\"a}ssle, Christian and Karlen, Walter},
      booktitle = {International Conference on Machine Learning},
      pages = {4525--4534},
      year = {2018},
      volume = {80},
      series = {Proceedings of Machine Learning Research},
      publisher = {PMLR},
      pdf = {http://proceedings.mlr.press/v80/schwab18a/schwab18a.pdf},
      url = {http://proceedings.mlr.press/v80/schwab18a.html},
    }


### Installation

- You can install the package and its dependencies using `pip install .` in the project's root directory.
- This project was designed for use with Python 2.7. We can not guarantee and have not tested compability with Python 3.

### Dataset

Please download the [Daphnet Freezing of Gait dataset at UCI](https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait) to run this code with an openly available dataset.

#### Usage

After installing the package's dependencies, you can use the provided example code to train a DSMT-Net on the above dataset using the following command (see the `dsmt_nets/parameters.py` file for descriptions of all the available parameters):

    python /path/to/dsmt_nets/dsmt_nets/train.py
    --dataset=/path/to/fog/dataset
    --num_epochs=300
    --output_directory=/path/to/your/output/directory
    --learning_rate=0.01
    --num_units=16
    --samples_per_segment=600

## Data Acknowledgement

We thank the authors of

    Bächlin, M., Plotnik, M., Roggen, D., Maidan, I., Hausdorff, J. M., Giladi, N., & Tröster, G. (2010). Wearable assistant for Parkinson's disease patients with the freezing of gait symptom. IEEE Trans. Information Technology in Biomedicine, 14(2), 436-446

for providing the Daphnet Freezing of Gait dataset to the public.

### Acknowledgements

This work was partially funded by the Swiss National Science Foundation (SNSF) project No. 167195 within the National Research Program (NRP) 75 “Big Data” and the Swiss Commission for Technology and Innovation (CTI) project No. 25531.
