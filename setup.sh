#!/usr/bin/env bash

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install numpy>=1.18
pip install pydantic==1.7.4
pip install pyserini>=0.16.0
pip install tensorboard==2.5.0
pip install transformers
pip install tokenizers>=0.10.2
pip install tqdm==4.58.0
pip install apex==0.1
pip install six==1.16.0
pip install datasets
pip install protobuf==3.20.1
