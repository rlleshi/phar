ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# ! https://github.com/NVIDIA/nvidia-docker/issues/1632
# currently image not working properly
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 \
    libxrender-dev libxext6 ffmpeg nano p7zip-full imagemagick wget unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
RUN git clone --recurse-submodules https://github.com/rlleshi/phar.git phar

# install mmaction, mmpose, mmdet
WORKDIR /workspace/phar/mmaction2
ENV FORCE_CUDA="1"
RUN pip install cython --no-cache-dir
RUN pip install --no-cache-dir -e .
WORKDIR /workspace/phar/mmpose
RUN pip install -r requirements.txt
RUN pip install -v -e .
RUN pip install mmdet==2.12.0

# install extra dependencies
WORKDIR /workspace/phar
RUN pip install -r requirements/extra.txt

# download models
RUN wget https://github.com/rlleshi/phar/releases/download/v1.0.0/audio.pth -O checkpoints/har/audio.pth \
    && wget https://github.com/rlleshi/phar/releases/download/v1.0.0/posec3d.pth -O checkpoints/har/posec3d.pth \
    && wget https://github.com/rlleshi/phar/releases/download/v1.0.0/timeSformer.pth -O checkpoints/har/timeSformer.pth \
    && wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth -O checkpoints/pose/hrnet_w32_coco_256x192.pth \
    && wget http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    -O checkpoints/detector/faster_rcnn_r50_fpn_1x_coco-person.pth
