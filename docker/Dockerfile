# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Nvidia Dockerfiles: https://github.com/NVIDIA-Omniverse/IsaacSim-dockerfiles
# Please check above link for license information.

# Base image
ARG ISAACSIM_VERSION
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

# Set default RUN shell to bash
SHELL ["/bin/bash", "-c"]

# Adds labels to the Dockerfile
LABEL version="1.0"
LABEL description="Dockerfile for building and running the Orbit framework inside Isaac Sim container image."

# Arguments
# Path to Isaac Sim root folder
ARG ISAACSIM_PATH
# Path to the Docker User Home
ARG DOCKER_USER_HOME
# Path to the Orbit directory
ARG ORBIT_USER_PATH
# non-root user name
ARG DOCKER_USER_NAME
# Orbit volume paths
ARG ORBIT_VOLUME_DOCS
ARG ORBIT_VOLUME_LOGS
ARG ORBIT_VOLUME_DATA
# User-specific paths
ARG USER_WORKSPACE

# Set environment variables
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE=/workspace
ENV GEOMETRYRL_PATH=/workspace/geometry_rl
ENV ORBIT_PATH=/workspace/orbit

# install Bazel
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
RUN mv bazel-archive-keyring.gpg /usr/share/keyrings
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

# Install dependencies and remove cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    bazel \
    patch \
    libglib2.0-0 \
    ncurses-term \
    vim \
    sudo \
    wget \
    unzip \
    git-lfs \
    openssh-server \
    tar \
    libarmadillo-dev \
    libnlopt-dev \
    libnlopt-cxx-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran \
    python3-pybind11 \
    && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Change ownership of the Isaac Sim root folder
# Note: this operation is time consuming, so it is done before copying the orbit directory
RUN chown -R ${DOCKER_USER_NAME}:${DOCKER_USER_NAME} /isaac-sim

# Add the default user to the sudo group
RUN usermod -aG sudo ${DOCKER_USER_NAME}

# Set passwordless sudo for the default user
RUN echo "${DOCKER_USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install git-lfs
RUN git lfs install

# Note: All copies happen here, as the chown happens just right after this
COPY ${USER_WORKSPACE}/orbit/ ${ORBIT_PATH}
COPY ${USER_WORKSPACE}/geometry_rl/requirements.txt ${WORKSPACE}/geometry_rl/requirements.txt
COPY ${USER_WORKSPACE}/others/ITPAL ${WORKSPACE}/others/ITPAL

# Change ownership of the workspace directory
RUN chown -R ${DOCKER_USER_NAME}:${DOCKER_USER_NAME} ${WORKSPACE}

# Set up a symbolic link between the installed Isaac Sim root folder and _isaac_sim in the orbit directory
RUN ln -sf ${ISAACSIM_PATH} ${ORBIT_PATH}/_isaac_sim
RUN cp -P ${ORBIT_PATH}/_isaac_sim ${GEOMETRYRL_PATH}/

# for singularity usage, have to create the directories that will binded
RUN mkdir -p ${ISAACSIM_PATH}/kit/cache && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/ov && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/pip && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/nvidia/GLCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nv/ComputeCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nvidia-omniverse/logs && \
    mkdir -p ${DOCKER_USER_HOME}/.local/share/ov/data && \
    mkdir -p ${DOCKER_USER_HOME}/Documents

# for singularity usage, create NVIDIA binary placeholders
RUN touch /bin/nvidia-smi && \
    touch /bin/nvidia-debugdump && \
    touch /bin/nvidia-persistenced && \
    touch /bin/nvidia-cuda-mps-control && \
    touch /bin/nvidia-cuda-mps-server && \
    touch /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && \
    touch /var/run/nvidia-persistenced/socket

# Switch to the non-root user
USER ${DOCKER_USER_NAME}

# installing Orbit dependencies
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ORBIT_PATH}/orbit.sh --install --extra

# aliasing orbit.sh and python for convenience
RUN echo "alias orbit=${ORBIT_PATH}/orbit.sh" >> ${HOME}/.bashrc && \
    echo "alias python=${ISAACSIM_PATH}/python.sh" >> ${HOME}/.bashrc && \
    echo "alias python3=${ISAACSIM_PATH}/python.sh" >> ${HOME}/.bashrc && \
    echo "alias pip='${ISAACSIM_PATH}/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias pip3='${ISAACSIM_PATH}/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias tensorboard='${ISAACSIM_PATH}/python.sh ${ISAACSIM_PATH}/tensorboard'" >> ${HOME}/.bashrc && \
    echo "export TZ=$(date +%Z)" >> ${HOME}/.bashrc

RUN echo "# Check and change folder ownership script" >> ${HOME}/.bashrc && \
    echo "folders=(" >> ${HOME}/.bashrc && \
    echo "    ${ORBIT_VOLUME_DOCS}" >> ${HOME}/.bashrc && \
    echo "    ${ORBIT_VOLUME_LOGS}" >> ${HOME}/.bashrc && \
    echo "    ${ORBIT_VOLUME_DATA}" >> ${HOME}/.bashrc && \
    echo "    ${ISAACSIM_PATH}/kit/cache" >> ${HOME}/.bashrc && \
    echo "    ${DOCKER_USER_HOME}/.cache" >> ${HOME}/.bashrc && \
    echo "    ${DOCKER_USER_HOME}/.nv" >> ${HOME}/.bashrc && \
    echo "    ${DOCKER_USER_HOME}/.nvidia-omniverse" >> ${HOME}/.bashrc && \
    echo "    ${ISAACSIM_PATH}/kit/logs/Kit/Isaac-Sim" >> ${HOME}/.bashrc && \
    echo "    ${DOCKER_USER_HOME}/.local/share/ov/data" >> ${HOME}/.bashrc && \
    echo "    ${DOCKER_USER_HOME}/Documents" >> ${HOME}/.bashrc && \
    echo ")" >> ${HOME}/.bashrc && \
    echo "for folder in \"\${folders[@]}\"; do" >> ${HOME}/.bashrc && \
    echo "    if [ -d \"\$folder\" ] && [ \"\$(stat -c '%U' \"\$folder\")\" = \"root\" ]; then" >> ${HOME}/.bashrc && \
    echo "        sudo chown -R ${DOCKER_USER_NAME}:${DOCKER_USER_NAME} \"\$folder\"" >> ${HOME}/.bashrc && \
    echo "    fi" >> ${HOME}/.bashrc && \
    echo "done" >> ${HOME}/.bashrc

###
# Install additional dependencies
###

RUN ${ORBIT_PATH}/orbit.sh -p -m pip install -U pip setuptools
RUN cd ${WORKSPACE}/others/ITPAL/cpp && CONDA_PREFIX="" ${ORBIT_PATH}/orbit.sh -p setup.py install
RUN ${ORBIT_PATH}/orbit.sh -p -m pip install \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ORBIT_PATH}/orbit.sh -p -m pip install -r ${WORKSPACE}/geometry_rl/requirements.txt \
    && ${ORBIT_PATH}/orbit.sh -p -m pip install typing_extensions==4.12.2

# make working directory as the geometry_rl directory
# this is the default directory when the container is run
WORKDIR ${GEOMETRYRL_PATH}
