FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
ENV TCNN_CUDA_ARCHITECTURES=86
ENV CUDA_HOME="/usr/local/cuda-11.6"
ENV PATH="$CUDA_HOME/bin:$PATH"
ENV LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
RUN apt-get update && apt-get install -y --no-install-recommends wget git ninja-build && \
    pip install packaging && \
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html && \
    pip install git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch && \
    git clone https://github.com/NVIDIA/apex /apex && cd /apex && git checkout 23.08 && python setup.py bdist_wheel --cpp_ext --cuda_ext && pip install dist/*.whl && \
    pip install opencv-python-headless pytorch-lightning==1.8 "pyglet<2" einops matplotlib kornia imageio dearpygui scipy pymcubes trimesh plotly
WORKDIR /volume