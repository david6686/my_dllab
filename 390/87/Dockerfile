FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
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
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list &&\
    APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="pip install  --no-cache-dir" && \
    GIT_CLONE="git clone --depth 1" && \
    CONDA="conda install -y" && \
    rm -rf  /var/lib/apt/lists/* \
            /etc/apt/sources.list.d/cuda.list \
            /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update  --fix-missing && \
    DEBIAN_FRONTEND=noninteractive  $APT_INSTALL software-properties-common && \
    add-apt-repository -y ppa:graphics-drivers/ppa && \
    add-apt-repository ppa:kelleyk/emacs &&\
    apt-get update \
    && \

# ==================================================================
# apt-get
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive  $APT_INSTALL \
        #語言包
        language-pack-en-base \
        language-pack-zh-hant \
        language-pack-zh-hant-base \
        bash \
        cpulimit \
        sudo \
        wget \
        bzip2 \
        nmon \
#         emacs25 \
        software-properties-common \
        ca-certificates \
        curl \
        cpulimit \
        dbus-x11 \
        fontconfig \
        git \
        rlwrap \
        silversearcher-ag \
        figlet \
        fish \
        cmake \
        screen \
        nano \
        pv \
        vim \
#         emacs \
        libjpeg-dev\
        libgl1-mesa-glx \
        libpng-dev \
        build-essential \
        unzip \
        zip \
        gzip \
        unrar \ 
        rar \
        tar \
        autojump \
        doxygen \
        firefox \
        htop \
        tmux \
        emacs25 \
        protobuf-compiler \
        libprotoc-dev \
        && \
    #setup emacs
#     add-apt-repository ppa:kelleyk/emacs &&\
#     apt-get update &&\
#     DEBIAN_FRONTEND=noninteractive  $APT_INSTALL emacs25 &&\
    mkdir -p /usr/local/share/fonts \
    && wget -qO- "${SCP_URL}" | tar xz -C /usr/local/share/fonts \
    && wget -q "${NNG_URL}" -P /usr/local/share/fonts \
    && fc-cache -fv \
    && \
    $GIT_CLONE  https://github.com/syl20bnr/spacemacs ~/.emacs.d \
        && \
    #setup autojump
    # echo 'source /usr/share/autojump/autojump.bash' >>~/.bash_profile && \
    echo 'source /usr/share/autojump/autojump.bash' >>~/.bash_profile \
        && \
# ==================================================================
# miniconda3
# ------------------------------------------------------------------ 
    #install miniconda3
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc \
    && \
# ==================================================================
# 設定顯示卡(for rancher)
# -----------------------------------------------------------------
    # apt-get update && apt-get install -y --no-install-recommends --allow-downgrades\
    #     libcudnn7=${CUDNN_VERSION} \
    #     libnccl2=${NCCL_VERSION} \
    #     libnccl-dev=${NCCL_VERSION} \
    #     && \
#     add-apt-repository -y ppa:graphics-drivers/ppa \
#     && \
#     apt-get update &&\
    # DEBIAN_FRONTEND=noninteractive  $APT_INSTALL \
    # nvidia-390 nvidia-390-dev libcuda1-390 nvidia-settings\
    # && \
# ==================================================================
# Install (pip) tensorflow keras pytorch
# ------------------------------------------------------------------
    $PIP_INSTALL \
    tensorflow-gpu==$TENSORFLOW_VERSION \
    keras \
    h5py \
    xmltodict \
    glances \
    nvidia-ml-py3 \
    jupyter \
    thefuck \
    psrecord \
    http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl  \
    torchvision \
    imgaug \
    onnx \
    cntk-gpu \
#     tensorflowjs \
    && \
# ================================================================== 
# intelpython-full 
# ------------------------------------------------------------------  
#     conda config --add channels intel\ 
#     && conda install  -y -q intelpython3_full=2018.0.3 python=3 \ 
#     && conda clean --all \ 
#     && apt-get update -qqq \ 
#     && apt-get install -y -q g++ \ 
# +
#     && apt-get autoremove \ 
#     && \ 
# ==================================================================
# Install (conda) theano sklearn scipy numpy ... ML package
# ------------------------------------------------------------------
    conda config --add channels intel \
    && \
    DEBIAN_FRONTEND=noninteractive $CONDA  \
    opencv \
    gensim \
    tqdm \
    dask \
    numpy \
    # jupyter notebook \
    scikit-learn \
    matplotlib \
    Cython \
    scipy \
    theano \
    protobuf \
    libprotobuf=3.2.0 \
    && \
    conda install -c conda-forge jupyterlab \
    && \
# ==================================================================
# boost
# ------------------------------------------------------------------

    # wget -O ~/boost.tar.gz https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz && \
    # tar -zxf ~/boost.tar.gz -C ~ && \
    # cd ~/boost_* && \
    # ./bootstrap.sh --with-python=python3.6 && \
    # ./b2 install --prefix=/usr/local && \
# ==================================================================
# Install Open MPI
# ------------------------------------------------------------------
    mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz && \
    tar zxf openmpi-3.0.0.tar.gz && \
    cd openmpi-3.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    cd ~ &&\
    rm -rf /tmp/openmpi \ 
    && \
# ==================================================================
# Install Horovod, temporarily using CUDA stubs
# ------------------------------------------------------------------
    ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod && \
    ldconfig \
    && \
# ==================================================================
# Create a wrapper for OpenMPI to allow running as root by default
# ------------------------------------------------------------------
    mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
    chmod a+x /usr/local/bin/mpirun \
    && \
# Configure OpenMPI to run good defaults:
# --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
    echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf \
    && \
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf \
    && \
# Install OpenSSH for MPI to communicate between containers
    DEBIAN_FRONTEND=noninteractive  $APT_INSTALL \ 
    openssh-client \
    openssh-server && \
    mkdir -p /var/run/sshd \
    && \


# Allow OpenSSH to talk to containers without asking for confirmation
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config  \
    && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    #fish setup
    sed -i -e "s/bin\/ash/usr\/bin\/fish/" /etc/passwd  && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    chmod +x /usr/bin/tini && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/* && \
    curl -Lo ~/.config/fish/functions/fisher.fish --create-dirs https://git.io/fisher &&\
    echo 'nvidia-smi' >>/root/.bashrc && \
    echo 'figlet "Wellcome"' >>/root/.bashrc && \
    echo 'source (conda info --root)/etc/fish/conf.d/conda.fish' >>~/.config/fish/config.fish && \
    #設定 matplotlib 在沒有gui環境下也能跑(backend設定為Agg)
    mkdir -p /root/.config/matplotlib && \
    echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
    
    

# Set up notebook config
# COPY jupyter_notebook_config.py /root/.jupyter/
# Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062
# COPY run_jupyter.sh /root/




# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006 8888

# WORKDIR "/root"
ENTRYPOINT [ "/usr/bin/tini", "--" ]
# CMD [ "/bin/bash" ]
CMD ["/usr/bin/fish"]
