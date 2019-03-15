FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
MAINTAINER Silentink (https://github.com/david6686/my_dllab)
ENV TENSORFLOW_VERSION=1.12.0
# ENV CUDNN_VERSION=7.0.5.15-1+cuda9.0
# ENV NCCL_VERSION=2.2.13-1+cuda9.0
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 
ENV PATH /opt/conda/bin:$PATH
ENV TINI_VERSION v0.16.1 
ENV SHELL /usr/bin/fish
#for conda
ENV PATH /opt/conda/envs/dllab/bin:$PATH
ENV CONDA_DEFAULT_ENV dllab
ENV CONDA_PREFIX /opt/conda/envs/dllab
ENV JAVA_HOME /docker-java-home/jre
# ENV UHOME="/home/emacs"
ENV BAZEL_VERSION 0.18.0
ENV JAVA_VERSION 8u191
ENV JAVA_DEBIAN_VERSION 8u191-b12-2ubuntu0.18.04.1
# Default fonts
ENV NNG_URL="https://github.com/google/fonts/raw/master/ofl/\
nanumgothic/NanumGothic-Regular.ttf" \
    SCP_URL="https://github.com/adobe-fonts/source-code-pro/\
archive/2.030R-ro/1.050R-it.tar.gz"
# tini
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
COPY TF_CONFIG.txt /TF_CONFIG.txt
##
ENV CATALINA_HOME /usr/local/tomcat
ENV PATH $CATALINA_HOME/bin:$PATH
RUN mkdir -p "$CATALINA_HOME"
WORKDIR $CATALINA_HOME

# let "Tomcat Native" live somewhere isolated
ENV TOMCAT_NATIVE_LIBDIR $CATALINA_HOME/native-jni-lib
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$TOMCAT_NATIVE_LIBDIR

# runtime dependencies for Tomcat Native Libraries
# Tomcat Native 1.2+ requires a newer version of OpenSSL than debian:jessie has available
# > checking OpenSSL library version >= 1.0.2...
# > configure: error: Your version of OpenSSL is not compatible with this version of tcnative
# see http://tomcat.10.x6.nabble.com/VOTE-Release-Apache-Tomcat-8-0-32-tp5046007p5046024.html (and following discussion)
# and https://github.com/docker-library/tomcat/pull/31
ENV OPENSSL_VERSION 1.1.0f-3+deb9u2

# ==================================================================
# startup setup
# ------------------------------------------------------------------
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list &&\
    #commend set
    APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="pip3 install  --no-cache-dir" && \
    GIT_CLONE="git clone --depth 1" && \
    CONDA="conda install -y" && \
    rm -rf  /var/lib/apt/lists/* \
            /etc/apt/sources.list.d/cuda.list \
            /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update  --fix-missing && \
    DEBIAN_FRONTEND=noninteractive  $APT_INSTALL software-properties-common && \
    add-apt-repository -y ppa:graphics-drivers/ppa && \   
    add-apt-repository ppa:deadsnakes/ppa && \ 
    # add-apt-repository ppa:kelleyk/emacs &&\
    apt-get update \
    && \
    # mkdir /docker-java-home && \
    # mkdir /docker-java-home/jre && \
    
# ==================================================================
# apt-get
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive  $APT_INSTALL \
        #語言包
        language-pack-en-base \
        language-pack-zh-hant \
        language-pack-zh-hant-base \
        #cuda
        # cuda-command-line-tools-10-0 \
        # cuda-cublas-dev-10-0 \
        # cuda-cudart-dev-10-0 \
        # cuda-cufft-dev-10-0 \
        # cuda-curand-dev-10-0 \
        # cuda-cusolver-dev-10-0 \
        # cuda-cusparse-dev-10-0 \
        #other
        bash\
        build-essential \
        bzip2 \
        ca-certificates \
        cmake \
        netbase \
        curl \
        gnupg \
        dirmngr \
        fontconfig \
        git \
        gzip \
        htop \
        libgl1-mesa-glx \
        libjpeg-dev\
        libpng-dev \
        libprotoc-dev \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        protobuf-compiler \
        rar \
        tmux \
        xz-utils \
        tar \
        sudo \
        unrar \ 
        unzip \
        vim \
        wget \
        zip \
        swig \
        openjdk-8-jdk \
        openjdk-8-jre \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        python3-pip \
        python3-opencv \
        # python3-setuptools\
        && \
        #clean
        apt-get clean && \
        wget -O ~/get-pip.py \
            https://bootstrap.pypa.io/get-pip.py && \
        python3.6 ~/get-pip.py && \
        ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
        ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    java -version &&\
# ==================================================================
# Install (pip) tensorflow keras pytorch
# ------------------------------------------------------------------
    $PIP_INSTALL \
    # 不可以用conda 因為conda 會cpu gpu 版都裝導致在用時找不到gpu
    # tensorflow-gpu==$TENSORFLOW_VERSION \ 
    setuptools \
    # pip install --upgrade pip \
    h5py \
    xmltodict \
    glances \
    nvidia-ml-py3 \
    scikit-learn \
    matplotlib \
    Cython \
    scipy \
    protobuf \
    tensorflow-gpu \
    # openc-python \
    keras \
    # http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl  \
    # torchvision \
    imgaug \
    onnx \
    # cntk-gpu \
#     tensorflowjs \
    && \
# ==================================================================
# set font
# ------------------------------------------------------------------
    mkdir -p /usr/local/share/fonts \
    && wget -qO- "${SCP_URL}" | tar xz -C /usr/local/share/fonts \
    && wget -q "${NNG_URL}" -P /usr/local/share/fonts \
    && fc-cache -fv \
    && \
# ==================================================================
# 設定顯示卡(for rancher)  (removed)
# -----------------------------------------------------------------
    # apt-get update && apt-get install -y --no-install-recommends --allow-downgrades\
    #     libcudnn7=${CUDNN_VERSION} \
    #     libnccl2=${NCCL_VERSION} \
    #     libnccl-dev=${NCCL_VERSION} \
    #     && \
    add-apt-repository -y ppa:graphics-drivers/ppa \
    && \
    apt-get update &&\
    DEBIAN_FRONTEND=noninteractive  $APT_INSTALL \
    nvidia-driver-410 nvidia-settings
    # nvidia-390 nvidia-390-dev libcuda1-390 nvidia-settings\
    # && \

# ==================================================================
# Install BAZEL & Install and Build Tensorflow
# ------------------------------------------------------------------
    # mkdir /bazel && \
    # wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    # wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    # chmod +x /bazel/installer.sh && \
    # /bazel/installer.sh && \
    # rm -f /bazel/installer.sh && \

    # git clone https://github.com/tensorflow/tensorflow.git --branch r1.12 --depth 1 && \
    # cd tensorflow && \
    # ls && \ 
    # ./configure < /TF_CONFIG.txt && \
    # bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package && \
    # ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg && \
    # pip install /tmp/tensorflow_pkg/*.whl &&\
    # cd ~ &&\
    # rm -rf tensorflow 
#########################################################################################分
#########################################################################################隔
#########################################################################################線
# add a simple script that can auto-detect the appropriate JAVA_HOME value
# based on whether the JDK or only the JRE is installed
RUN { \
        echo '#!/bin/sh'; \
        echo 'set -e'; \
        echo; \
        echo 'dirname "$(dirname "$(readlink -f "$(which javac || which java)")")"'; \
    } > /usr/local/bin/docker-java-home \
    && chmod +x /usr/local/bin/docker-java-home
# do some fancy footwork to create a JAVA_HOME that's cross-architecture-safe
RUN ln -svT "/usr/lib/jvm/java-8-openjdk-$(dpkg --print-architecture)" /docker-java-home

RUN set -ex; \
    currentVersion="$(dpkg-query --show --showformat '${Version}\n' openssl)"; \
    if dpkg --compare-versions "$currentVersion" '<<' "$OPENSSL_VERSION"; then \
        if ! grep -q stretch /etc/apt/sources.list; then \
# only add stretch if we're not already building from within stretch
            { \
                echo 'deb http://deb.debian.org/debian stretch main'; \
                echo 'deb http://security.debian.org stretch/updates main'; \
                echo 'deb http://deb.debian.org/debian stretch-updates main'; \
            } > /etc/apt/sources.list.d/stretch.list; \
            { \
# add a negative "Pin-Priority" so that we never ever get packages from stretch unless we explicitly request them
                echo 'Package: *'; \
                echo 'Pin: release n=stretch*'; \
                echo 'Pin-Priority: -10'; \
                echo; \
# ... except OpenSSL, which is the reason we're here
                echo 'Package: openssl libssl*'; \
                echo "Pin: version $OPENSSL_VERSION"; \
                echo 'Pin-Priority: 990'; \
            } > /etc/apt/preferences.d/stretch-openssl; \
        fi; \
        apt-get update; \
        apt-get install -y --no-install-recommends openssl="$OPENSSL_VERSION"; \
        rm -rf /var/lib/apt/lists/*; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
        libapr1 \
    && rm -rf /var/lib/apt/lists/*

# see https://www.apache.org/dist/tomcat/tomcat-$TOMCAT_MAJOR/KEYS
# see also "update.sh" (https://github.com/docker-library/tomcat/blob/master/update.sh)

# ENV GPG_KEYS 05AB33110949707C93A279E3D3EFE6B686867BA6 07E48665A34DCAFAE522E5E6266191C37C037D42 47309207D818FFD8DCD3F83F1931D684307A10A5 541FBE7D8F78B25E055DDEE13C370389288584E7 61B832AC2F1C5A90F0F9B00A1C506407564C17A3 713DA88BE50911535FE716F5208B0AB1D63011C7 79F7026C690BAA50B92CD8B66A3AD3F4F22C4FED 9BA44C2621385CB966EBA586F72C284D731FABEE A27677289986DB50844682F8ACB77FC2E86E29AC A9C5DF4D22E99998D9875A5110C01C5A2F6059E7 DCFD35E0BF8CA7344752DE8B6FB21E8933C60243 F3A04C595DB5B6A5F1ECA43E3B7BBB100D811BBE F7DA48BB64BCB84ECBA7EE6935CD23C10D498E23
ENV GPG_KEYS 05AB33110949707C93A279E3D3EFE6B686867BA6 07E48665A34DCAFAE522E5E6266191C37C037D42 47309207D818FFD8DCD3F83F1931D684307A10A5 541FBE7D8F78B25E055DDEE13C370389288584E7 61B832AC2F1C5A90F0F9B00A1C506407564C17A3 713DA88BE50911535FE716F5208B0AB1D63011C7 79F7026C690BAA50B92CD8B66A3AD3F4F22C4FED 9BA44C2621385CB966EBA586F72C284D731FABEE A27677289986DB50844682F8ACB77FC2E86E29AC A9C5DF4D22E99998D9875A5110C01C5A2F6059E7 DCFD35E0BF8CA7344752DE8B6FB21E8933C60243 F3A04C595DB5B6A5F1ECA43E3B7BBB100D811BBE F7DA48BB64BCB84ECBA7EE6935CD23C10D498E23

# ENV GPG_KEYS 05AB33110949707C93A279E3D3EFE6B686867BA607E48665A34DCAFAE522E5E6266191C37C037D4247309207D818FFD8DCD3F83F1931D684307A10A5541FBE7D8F78B25E055DDEE13C370389288584E761B832AC2F1C5A90F0F9B00A1C506407564C17A3713DA88BE50911535FE716F5208B0AB1D63011C779F7026C690BAA50B92CD8B66A3AD3F4F22C4FED9BA44C2621385CB966EBA586F72C284D731FABEEA27677289986DB50844682F8ACB77FC2E86E29ACA9C5DF4D22E99998D9875A5110C01C5A2F6059E7DCFD35E0BF8CA7344752DE8B6FB21E8933C60243F3A04C595DB5B6A5F1ECA43E3B7BBB100D811BBEF7DA48BB64BCB84ECBA7EE6935CD23C10D498E23

ENV TOMCAT_MAJOR 8
ENV TOMCAT_VERSION 8.5.38
ENV TOMCAT_SHA512 3a3e624014faf87091e6dbb8bad13c68240955d62301d22cf3d75a1766859dd97500d6850fbd5d3dc012f08f9301eb24c24fa7175bcca616767fa5c18875072d

ENV TOMCAT_TGZ_URLS \
# https://issues.apache.org/jira/browse/INFRA-8753?focusedCommentId=14735394#comment-14735394
    https://www.apache.org/dyn/closer.cgi?action=download&filename=tomcat/tomcat-$TOMCAT_MAJOR/v$TOMCAT_VERSION/bin/apache-tomcat-$TOMCAT_VERSION.tar.gz \
# if the version is outdated, we might have to pull from the dist/archive :/
    https://www-us.apache.org/dist/tomcat/tomcat-$TOMCAT_MAJOR/v$TOMCAT_VERSION/bin/apache-tomcat-$TOMCAT_VERSION.tar.gz \
    https://www.apache.org/dist/tomcat/tomcat-$TOMCAT_MAJOR/v$TOMCAT_VERSION/bin/apache-tomcat-$TOMCAT_VERSION.tar.gz \
    https://archive.apache.org/dist/tomcat/tomcat-$TOMCAT_MAJOR/v$TOMCAT_VERSION/bin/apache-tomcat-$TOMCAT_VERSION.tar.gz

ENV TOMCAT_ASC_URLS \
    https://www.apache.org/dyn/closer.cgi?action=download&filename=tomcat/tomcat-$TOMCAT_MAJOR/v$TOMCAT_VERSION/bin/apache-tomcat-$TOMCAT_VERSION.tar.gz.asc \
# not all the mirrors actually carry the .asc files :'(
    https://www-us.apache.org/dist/tomcat/tomcat-$TOMCAT_MAJOR/v$TOMCAT_VERSION/bin/apache-tomcat-$TOMCAT_VERSION.tar.gz.asc \
    https://www.apache.org/dist/tomcat/tomcat-$TOMCAT_MAJOR/v$TOMCAT_VERSION/bin/apache-tomcat-$TOMCAT_VERSION.tar.gz.asc \
    https://archive.apache.org/dist/tomcat/tomcat-$TOMCAT_MAJOR/v$TOMCAT_VERSION/bin/apache-tomcat-$TOMCAT_VERSION.tar.gz.asc

RUN set -eux; \
    \
    savedAptMark="$(apt-mark showmanual)"; \
    apt-get update; \
    \
    apt-get install -y --no-install-recommends gnupg dirmngr; \
    \
    export GNUPGHOME="$(mktemp -d)"; \
    for key in $GPG_KEYS; do \
        gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys "$key"; \
        # gpg --keyserver ha.pool.sks-keyservers.net --recv-keys "$key"; \
    done; \
    \
    apt-get install -y --no-install-recommends wget ca-certificates; \
    \
    success=; \
    for url in $TOMCAT_TGZ_URLS; do \
        if wget -O tomcat.tar.gz "$url"; then \
            success=1; \
            break; \
        fi; \
    done; \
    [ -n "$success" ]; \
    \
    echo "$TOMCAT_SHA512 *tomcat.tar.gz" | sha512sum -c -; \
    \
    success=; \
    for url in $TOMCAT_ASC_URLS; do \
        if wget -O tomcat.tar.gz.asc "$url"; then \
            success=1; \
            break; \
        fi; \
    done; \
    [ -n "$success" ]; \
    \
    gpg --batch --verify tomcat.tar.gz.asc tomcat.tar.gz; \
    tar -xvf tomcat.tar.gz --strip-components=1; \
    rm bin/*.bat; \
    rm tomcat.tar.gz*; \
    command -v gpgconf && gpgconf --kill all || :; \
    rm -rf "$GNUPGHOME"; \
    \
    nativeBuildDir="$(mktemp -d)"; \
    tar -xvf bin/tomcat-native.tar.gz -C "$nativeBuildDir" --strip-components=1; \
    apt-get install -y --no-install-recommends \
        dpkg-dev \
        gcc \
        libapr1-dev \
        libssl-dev \
        make \
        "openjdk-${JAVA_VERSION%%[.~bu-]*}-jdk=$JAVA_DEBIAN_VERSION" \
    ; \
    ( \
        export CATALINA_HOME="$PWD"; \
        cd "$nativeBuildDir/native"; \
        gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"; \
        ./configure \
            --build="$gnuArch" \
            --libdir="$TOMCAT_NATIVE_LIBDIR" \
            --prefix="$CATALINA_HOME" \
            --with-apr="$(which apr-1-config)" \
            --with-java-home="$(docker-java-home)" \
            --with-ssl=yes; \
        make -j "$(nproc)"; \
        make install; \
    ); \
    rm -rf "$nativeBuildDir"; \
    rm bin/tomcat-native.tar.gz; \
    \
# reset apt-mark's "manual" list so that "purge --auto-remove" will remove all build dependencies
    apt-mark auto '.*' > /dev/null; \
    [ -z "$savedAptMark" ] || apt-mark manual $savedAptMark; \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false; \
    rm -rf /var/lib/apt/lists/*; \
    \
# sh removes env vars it doesn't support (ones with periods)
# https://github.com/docker-library/tomcat/issues/77
    find ./bin/ -name '*.sh' -exec sed -ri 's|^#!/bin/sh$|#!/usr/bin/env bash|' '{}' +; \
    \
# fix permissions (especially for running as non-root)
# https://github.com/docker-library/tomcat/issues/35
    chmod -R +rX .; \
    chmod 777 logs work

# verify Tomcat Native is working properly
RUN set -e \
    && nativeLines="$(catalina.sh configtest 2>&1)" \
    && nativeLines="$(echo "$nativeLines" | grep 'Apache Tomcat Native')" \
    && nativeLines="$(echo "$nativeLines" | sort -u)" \
    && if ! echo "$nativeLines" | grep 'INFO: Loaded APR based Apache Tomcat Native library' >&2; then \
        echo >&2 "$nativeLines"; \
        exit 1; \
    fi

EXPOSE 8080
CMD ["catalina.sh", "run"]