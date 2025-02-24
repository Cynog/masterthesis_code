FROM gptdev/shell

# gpt quiet import
ENV GPT_QUIET=YES

# switch back to root user
USER root

# set timezone needed for some package installations
ARG DEBIAN_FRONTEND=noninteractive 
ENV TZ=UTC

# install system packages
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN apt-get update && apt-get install -y \
    git \
    fish \
    git-lfs \
    gcc \
    python3-dev \
    pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# set fish as default shell for gpt user
RUN chsh -s /usr/bin/fish gpt

# switch back to gpt user
USER gpt
WORKDIR /home/gpt

# install python packages
COPY .devcontainer/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# install qcd_ml python package with torch cpu only version
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir qcd_ml@git+https://github.com/daknuett/qcd_ml@main
