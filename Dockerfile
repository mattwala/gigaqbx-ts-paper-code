FROM debian:stretch
LABEL maintainer="wala1@illinois.edu"

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential xzdec wget curl ca-certificates texlive-base texlive-pictures texlive-science \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash inteq
USER inteq

# Because there's a file with a space in its name
ENV QUOTING_STYLE=literal

# Install latexrun script
RUN mkdir /home/inteq/bin
WORKDIR /home/inteq/bin
RUN curl -O https://raw.githubusercontent.com/aclements/latexrun/566809a5b8feecc16a13033f3036b6a3c84e8ed7/latexrun \
    && chmod +x latexrun
ENV PATH="/home/inteq/bin:${PATH}"

# Install Conda environment
RUN mkdir /home/inteq/gigaqbx-ts-results
WORKDIR /home/inteq/gigaqbx-ts-results
COPY --chown=inteq . .
RUN ./install.sh \
    && .miniconda3/bin/conda clean --all --yes

# Install extra LaTeX bits
RUN tlmgr init-usertree \
    && tlmgr install etoolbox multirow
