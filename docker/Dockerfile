FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

MAINTAINER Lukas Liebel <lukas.liebel@tum.de>

# Install utils
# RUN apt update && apt install -y \
#   tmux \
#   htop \
#   wget \
#   nano \
#   git \
#   rsync
RUN pip install --upgrade pip

# Install required packages
RUN pip install cython
RUN pip install matplotlib
RUN pip install numpy
RUN pip install scipy
RUN pip install tqdm
RUN pip install visdom
RUN pip install pydensecrf
RUN pip install protobuf
RUN pip install tensorboardX
RUN pip install tensorflow
RUN pip install tensorboard
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install imageio

# Setup matplotlib
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

ENV workdir /root
WORKDIR ${workdir}

CMD bash
