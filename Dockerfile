#Ubuntu image
FROM ubuntu:16.04

# FROM defines the base image
FROM nvidia/cuda:10.0-base
FROM nvidia/cuda:10.0-cudnn7-runtime
FROM nvidia/cuda:10.0-cudnn7-devel

RUN apt-get update && \
    apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.5 && \
    apt-get update -y

RUN apt-get update && apt-get install -y \
    --no-install-recommends python3.5 python3.5-dev \
    python3-pip && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >>/root/.bash_aliases
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
#set the working directory
WORKDIR /mask_rcnn

ADD mask_rcnn_coco.h5 .
ADD mrcnn/ ./mrcnn
ADD samples/ ./samples
ADD tracker/ ./tracker
RUN mkdir videos

#ADD cafe.wmv /tf_worker
#ADD object_detection /tf_worker/object_detection
#ADD object_det_0mq.py /tf_worker


#RUN apt-get update && apt-get -y install python-pip python-dev 
#RUN pip install pip -U
RUN pip install zmq requests Pillow Cython numpy
RUN pip install keras opencv-python h5py imgaug ipython
RUN pip install pycocotools scikit-image scikit-learn matplotlib 
RUN pip install --upgrade tensorflow-gpu==tf-nightly-gpu
WORKDIR /mask_rcnn/samples

CMD ["python3.5", "segmt-live-video-zmq.py"]
