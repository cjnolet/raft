FROM gitlab-master.nvidia.com:5005/legate/quickstart.internal/legate-selene-ucx:2023-05-30-155736

RUN source activate legate && \
  conda install -y -c conda-forge mamba && \
  mamba install -y -c conda-forge scikit-learn scipy numpy && \
  mamba install -y -c conda-forge -c rapidsai -c nvidia pylibraft=23.06* raft-dask=23.06*

RUN apt-get update && apt-get install -y git openssh-server
RUN git clone https://github.com/nv-legate/legate.core.git /usr/local/legate.core

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub

ADD . /opt/legate/raft-legate
WORKDIR /opt/legate/raft-legate

RUN echo $(ls -la /opt/conda/envs/legate/lib/cmake)

RUN source activate legate && \
    export CONDA_PREFIX=/opt/conda/envs/legate && \
    export CMAKE_INSTALL_PREFIX=$CONDA_PREFIX && \
    bash -x '/opt/legate/raft-legate/build.sh'
