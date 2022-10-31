#!/usr/bin/env bash

# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pytorch torchvision torchaudio -c pytorch-nightly
pip install numpy
pip install transformers
pip install tokenizers
pip install tqdm
pip install datasets
pip install protobuf==3.20.1
