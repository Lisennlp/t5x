#!/bin/bash

sudo apt update

python3 -m pip install -U pip setuptools wheel ipython
python3 -m pip install --upgrade pip
python3 -m pip install https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20230724-py3-none-any.whl

pip install git+https://github.com/google-research/jestimator
pip install protobuf==3.20.3

git clone --branch=main https://github.com/Lisennlp/t5x.git
cd ~/t5x
pip install -e .


cd ~
pip install flax==0.7.1

echo y | python3 -m pip uninstall t5[gcp]
echo y | python3 -m pip uninstall t5

git clone --branch=main https://github.com/XueFuzhao/text-to-text-transfer-transformer.git
cd text-to-text-transfer-transformer
python3 setup.py install

echo y | python3 -m pip uninstall seqio
echo y | python3 -m pip uninstall seqio-nightly
git clone  --branch=main https://github.com/XueFuzhao/seqio.git
cd seqio
python3 setup.py install

cd ~
git clone  --branch=main https://github.com/Lisennlp/flaxformer.git
cd flaxformer
pip install -e .

python3 -m pip install gast
python3 -m pip install astunparse
python3 -m pip install flatbuffers
python3 -m pip install tensorboard
python3 -m pip install keras
python3 -m pip install tensorflow_estimator
python3 -m pip install libcst
python3 -m pip install portalocker
python3 -m pip install tabulate
python3 -m pip install colorama
python3 -m pip install lxml
python3 -m pip install joblib
python3 -m pip install threadpoolctl
python3 -m pip install tfds-nightly==4.6.0.dev202210040045
# python3 -m pip install tensorflow-datasets==4.3.0
python3 -m pip install h5py

cd ~
git clone https://github.com/google/aqt.git
cd aqt
python3 setup.py install



pip install -f https://storage.googleapis.com/jax-releases/libtpu_releases.html jax[tpu]==0.4.16
pip install tf_keras
# tensorflow_datasets

