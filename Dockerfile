FROM nvidia/cuda:9.0-devel-ubuntu16.04 
# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully 
ENV TENSORFLOW_VERSION=1.6.0 
ENV CUDNN_VERSION=7.0.5.15-1+cuda9.0 
ENV NCCL_VERSION=2.1.15-1+cuda9.0 
# Python 2.7 or 3.5 is supported by Ubuntu Xenial out of the box 
ENV PYTHON3_VERSION=3.5 
ENV PYTHON2_VERSION=2.7
ENV PATH /opt/conda/bin:$PATH
ARG CAFFE_VERSION=master
ARG THEANO_VERSION=master
# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# torch         latest (git)
# chainer       latest (pip)
# cntk          2.5.1  (pip)
# pytorch       0.4.0  (pip)
# tensorflow    latest (pip)
# theano        1.0.1  (git)
# keras         latest (pip)
# opencv        3.4.1  (git)
# ==================================================================

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        cmake \
        git \
        curl \
        vim \
        wget \
        unzip \
        ca-certificates \
        libcudnn7=$CUDNN_VERSION \
        libnccl2=$NCCL_VERSION \
        libnccl-dev=$NCCL_VERSION \
#         libgtk2.0 \
#         libjpeg-dev \
#         libpng-dev \
#         libffi-dev \
# 		libfreetype6-dev \
# 		libhdf5-dev \
# 		libjpeg-dev \
# 		liblcms2-dev \
# 		libopenblas-dev \
# 		liblapack-dev \
# 		libpng12-dev \
# 		libssl-dev \
# 		libtiff5-dev \
# 		libwebp-dev \
#         libzmq3-dev \
        software-properties-common \
        python$PYTHON3_VERSION \
        python$PYTHON3_VERSION-dev \
        python$PYTHON3_VERSION-tk \
        python$PYTHON2_VERSION \
        python$PYTHON2_VERSION-dev \
        python$PYTHON2_VERSION-tk \
#         zlib1g-dev \
# 		qt5-default \
# 		libvtk6-dev \
# 		zlib1g-dev \
# 		libjpeg-dev \
# 		libwebp-dev \
# 		libpng-dev \
# 		libtiff5-dev \
# 		libjasper-dev \
# 		libopenexr-dev \
# 		libgdal-dev \
# 		libdc1394-22-dev \
# 		libavcodec-dev \
# 		libavformat-dev \
# 		libswscale-dev \
# 		libtheora-dev \ 
# 		libvorbis-dev \
# 		libxvidcore-dev \
# 		libx264-dev \
#         libatlas-base-dev \
#         libgflags-dev \
#         libgoogle-glog-dev \
#         libhdf5-serial-dev \
#         libleveldb-dev \
#         liblmdb-dev \
#         libprotobuf-dev \
#         libsnappy-dev \
#         protobuf-compiler \
#         libboost-all-dev \
# 		libgflags-dev \
# 		libgoogle-glog-dev \
# 		libhdf5-serial-dev \
# 		libleveldb-dev \
# 		liblmdb-dev \
        libopencv-dev \
		yasm \
		libopencore-amrnb-dev \
		libopencore-amrwb-dev \
		libv4l-dev \
		libxine2-dev \
		libtbb-dev \
        libeigen3-dev \
        emacs \
        ant \
        doxygen \
        python-setuptools \
        fish \
        python-numpy \
		python-scipy \
		python-nose \
		python-h5py \
		python-skimage \
		python-matplotlib \
		python-pandas \
		python-sklearn \
        python-sympy \
		
        && \
        apt-get clean && \
        apt-get autoremove && \ 
        # Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
        update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3  
RUN git clone https://github.com/syl20bnr/spacemacs ~/.emacs.d &&\
    chsh -s /usr/bin/fish
# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
    rm get-pip.py   
# Install other useful Python packages using pip
RUN pip --no-cache-dir install --upgrade ipython && \
	pip --no-cache-dir install \
		Cython \
		ipykernel \
		jupyter \
		path.py \
		Pillow \
		pygments \
		six \
		sphinx \
		wheel \
		zmq \
		&& \
        python -m ipykernel.kernelspec
# ==================================================================
# pytorch
# ------------------------------------------------------------------
RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl  
RUN pip3 --no-cache-dir install torchvision
# RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     /opt/conda/bin/conda clean -tipsy && \
#     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#     echo "conda activate base" >> ~/.bashrc
# ==================================================================
# tensorflow  keras h5py
# ------------------------------------------------------------------
RUN pip install --no-cache-dir tensorflow-gpu==$TENSORFLOW_VERSION keras h5py
# ==================================================================
# Caffe
# ------------------------------------------------------------------
RUN git clone -b ${CAFFE_VERSION} --depth 1 https://github.com/BVLC/caffe.git /root/caffe && \
	cd /root/caffe && \
	cat python/requirements.txt | xargs -n1 pip install && \
	mkdir build && cd build && \
	cmake -DUSE_CUDNN=1 -DBLAS=Open .. && \
	make -j"$(nproc)" all && \
	make install

# Set up Caffe environment variables
ENV CAFFE_ROOT=/root/caffe
ENV PYCAFFE_ROOT=$CAFFE_ROOT/python
ENV PYTHONPATH=$PYCAFFE_ROOT:$PYTHONPATH \
	PATH=$CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH

RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
# ==================================================================
# THEANO
# ------------------------------------------------------------------
# Install Theano and set up Theano config (.theanorc) for CUDA and OpenBLAS
RUN pip install --no-cache-dir --upgrade git+git://github.com/Theano/Theano.git@${THEANO_VERSION} && \
	\
	echo "[global]\ndevice=gpu\nfloatX=float32\noptimizer_including=cudnn\nmode=FAST_RUN \
		\n[lib]\ncnmem=0.95 \
		\n[nvcc]\nfastmath=True \
		\n[blas]\nldflag = -L/usr/lib/openblas-base -lopenblas \
		\n[DebugMode]\ncheck_finite=1" \
    > /root/.theanorc
# ==================================================================
# OPEN MPI
# ------------------------------------------------------------------
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz && \
    tar zxf openmpi-3.0.0.tar.gz && \
    cd openmpi-3.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# ==================================================================
# Install Horovod, temporarily using CUDA stubs 
# ------------------------------------------------------------------
RUN ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod && \
    ldconfig
# Create a wrapper for OpenMPI to allow running as root by default 
RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
    chmod a+x /usr/local/bin/mpirun
# Configure OpenMPI to run good defaults: #   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0 
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf
# Set default NCCL parameters 
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf
# Install OpenSSH for MPI to communicate between containers 
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd
# Allow OpenSSH to talk to containers without asking for confirmation 
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
# ==================================================================
# cntk
# ------------------------------------------------------------------
RUN pip https://cntk.ai/PythonWheel/GPU/cntk_gpu-2.5.1-cp36-cp36m-linux_x86_64.whl \
# ==================================================================
# opencv 
# ------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/opencv/opencv.git /root/opencv && \
	cd /root/opencv && \
	mkdir build && \
	cd build && \
	cmake -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON .. && \
	make -j"$(nproc)"  && \
	make install && \
	ldconfig && \
	echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc


# Set up notebook config
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062
COPY run_jupyter.sh /root/

# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006 8888

WORKDIR "/root"
# ==================================================================
# ENTRYPOINT
# ------------------------------------------------------------------

# ENTRYPOINT nvidia-smi && fish
CMD nvidia-smi && fish
