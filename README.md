# Coding Kendall's Shape Trajectories for 3D Action Recognition 

This repository provides the official PyTorch and Keras implementation of the CVPR 2018 paper: [Coding Kendall's Shape Trajectories for 3D Action Recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Tanfous_Coding_Kendalls_Shape_CVPR_2018_paper.html).


## Overview

> 3D skeletal data is naturally represented as a sequence of shapes evolving over time. These sequences lie on non-linear manifolds, making standard machine learning approaches less effective. This work introduces a novel approach for encoding such trajectories using:

- Kendall's shape space geometry
- Intrinsic sparse coding using convex optimization
- Temporal encoding using either:
    - Bi-directional LSTMs (learned end-to-end)
    - Fourier Temporal Pyramid with linear SVM

This manifold-aware pipeline yields sparse, discriminative, and vector-space-compatible representations for human action recognition.


## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Preprocessing and Sparse Coding
The data is expected in the format of centered and scaled 3D joint positions per frame. To compute sparse codes for each shape sequence:
```bash
from sparse_coding import sparse_coding
# Provide list of shape sequences and a learned dictionary
sparse_coding(sequences, dictionary, lam, output_dir)
```

You can use sparse_coding_parallel() for multiprocessing.

### LSTM Classification
Train and evaluate a bidirectional LSTM on the sparse codes:

```bash
from lstm_model import bi_lstm
accuracy, scores = bi_lstm(opt, data, subject_labels, action_labels)
```

Ensure opt is a `config` object with fields like:
- `n_classes`
- `train_subjects`
- `lstm_size`
- `dropout_prob`
- `nb_epochs`
- `b_size`


## Citation 
If you use this codebase in your research, please cite:

``` 
@inproceedings{ben2018coding,
  title={Coding Kendall's shape trajectories for 3D action recognition},
  author={Ben Tanfous, Amor and Drira, Hassen and Ben Amor, Boulbaba},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2840--2849},
  year={2018}
}
```
