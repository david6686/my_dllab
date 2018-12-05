FROM silentink:dllab:base
# FROM nvidia/cuda:9.0-devel-ubuntu16.04
MAINTAINER Silentink (https://github.com/david6686/my_dllab)
# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (conda)
# jupyter       latest (pip)
# pytorch       latest  (pip)
# tensorflow    1.8.0 (pip)
# tensorflow-gpu    1.8.0 (pip)
# tensorflowjs    1.8.0 (latest)
# theano        1.0.1  (conda)
# keras         latest (pip)
# opencv        latest  (conda)
# tensorflow.js latest (pip)
# onnx          latest (pip)
# cntk          latest (pip)
# ==================================================================
ENV TENSORFLOW_VERSION=1.8.0
# ENV CUDNN_VERSION=7.0.5.15-1+cuda9.0
# ENV NCCL_VERSION=2.2.13-1+cuda9.0
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 
ENV PATH /opt/conda/bin:$PATH
ENV TINI_VERSION v0.16.1 
ENV SHELL /usr/bin/fish
# ENV UHOME="/home/emacs"
# Default fonts
ENV NNG_URL="https://github.com/google/fonts/raw/master/ofl/\
nanumgothic/NanumGothic-Regular.ttf" \
    SCP_URL="https://github.com/adobe-fonts/source-code-pro/\
archive/2.030R-ro/1.050R-it.tar.gz"
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# ==================================================================
# startup setup
# ------------------------------------------------------------------
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="pip install  --no-cache-dir" && \
    GIT_CLONE="git clone --depth 1" && \
    CONDA="conda install -y" && \
    rm -rf  /var/lib/apt/lists/* \
            /etc/apt/sources.list.d/cuda.list \
            /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update  --fix-missing && \
    DEBIAN_FRONTEND=noninteractive  $APT_INSTALL \
    nvidia-390 nvidia-390-dev libcuda1-390 nvidia-settings
