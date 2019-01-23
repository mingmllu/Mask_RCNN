#Ubuntu image
FROM ubuntu:16.04

# FROM defines the base image
FROM nvidia/cuda:9.0-base
FROM nvidia/cuda:9.0-cudnn7-runtime
FROM nvidia/cuda:9.0-cudnn7-devel

RUN apt-get update && \
    apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6 && \
    apt-get update -y

RUN apt-get update && apt-get install -y \
    --no-install-recommends python3.6 python3-pip && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >>/root/.bash_aliases

#set the working directory
WORKDIR /mask_rcnn

ADD mask_rcnn_coco.h5 .
ADD mrcnn/ .
ADD samples/ .

#ADD cafe.wmv /tf_worker
#ADD object_detection /tf_worker/object_detection
#ADD object_det_0mq.py /tf_worker


#RUN apt-get update && apt-get -y install python-pip python-dev 
#RUN pip install pip -U
RUN pip install zmq requests Pillow
RUN pip install --upgrade tensorflow-gpu

WORKDIR /mask_rcnn/samples

CMD ["python", "segmt-live-video-save-masked-frames.py"]
