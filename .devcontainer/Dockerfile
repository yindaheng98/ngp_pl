FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
COPY requirements.txt /requirements.txt
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y --no-install-recommends wget git && \
    wget -O /cuda-keyring_1.0-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i /cuda-keyring_1.0-1_all.deb
ENV TCNN_CUDA_ARCHITECTURES=86
RUN pip install packaging && \
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html && \
    pip install git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch && \
    git clone https://github.com/NVIDIA/apex /apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" /apex && \
    pip install -r /requirements.txt && pip install opencv-python-headless torchmetrics==0.9.3
WORKDIR /volume