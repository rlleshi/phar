ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y nano p7zip-full imagemagick unrar wget unzip

# Install mmcv-full
RUN pip install mmcv-full==1.3.11 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

# Install MMAction2
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmaction2.git /mmaction2
WORKDIR /mmaction2
RUN mkdir -p /mmaction2/data
ENV FORCE_CUDA="1"
RUN pip install cython --no-cache-dir
RUN pip install --no-cache-dir -e .

RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose
RUN pip install -r requirements.txt
RUN pip install -v -e .  # or "python setup.py develop"

WORKDIR /mmaction2
