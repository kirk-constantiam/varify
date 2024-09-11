FROM gitpod/workspace-python:2023-01-16-03-31-28

# Reinstall Python with --enable-shared for reticulate
ARG PYTHON_VERSION=3.8.14
RUN rm -rf ${HOME}/.pyenv/versions/${PYTHON_VERSION}
RUN PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}

USER root

# Install R
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc |\
    tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc && \
    add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

RUN apt-get update && \
    apt-get install -y gdebi-core r-base r-base-dev s3fs lftp graphviz graphviz-dev

# Install RStudio
RUN wget https://download2.rstudio.org/server/bionic/amd64/rstudio-server-2022.07.2-576-amd64.deb
RUN gdebi -n rstudio-server-2022.07.2-576-amd64.deb
RUN rm -rf rstudio-server-2022.07.2-576-amd64.deb

# Base R packages
RUN R -e 'install.packages(c("tidyverse", "BiocManager"))'
RUN R -e 'install.packages(c("languageserver", "shiny", "reticulate", "remotes", "kableExtra", "ggrepel"))'
RUN R -e 'BiocManager::install(c("EnsDb.Hsapiens.v86", "qval"))'

# Reset gitpod user credentials (for Rstudio login)
RUN echo "gitpod:gitpod" | chpasswd
USER gitpod

## Configure RStudio to start from the varify directory
RUN echo "session-default-working-dir=/workspace/varify" | sudo tee -a /etc/rstudio/rsession.conf && \
    echo "session-default-new-project-dir=/workspace/varify" | sudo tee -a /etc/rstudio/rsession.conf

# Base Python packages
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    polars \
    scipy \
    matplotlib \
    seaborn \
    plotly \
    "pymc>=5" \
    scanpy \
    jupyterlab \
    rpy2 \
    pygraphviz \
    awscli \
    colour \
    pyscenic \
    adjustText
