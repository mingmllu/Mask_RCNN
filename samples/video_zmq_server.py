#!/usr/bin/env python

import numpy as np

import time
import sys
import os
import zmq
import json
import urllib
import imutils
from imutils.video import FileVideoStream
import logging
import cv2

SERVICE_SOCKET = "tcp://*:5566"

log=logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main(service_socket, image_file_name):

  #read events from stdin and push results on stdout

  # service_socket: in the format tcp://*:port
  
  context=zmq.Context()
  socket = context.socket(zmq.REP)
  # ZMQ server must be listening on request first
  socket.bind(service_socket)

  print("[INFO] starting video file thread...")
  fvs = FileVideoStream(image_file_name).start()
  time.sleep(1.0)

  while fvs.more():
    frame = fvs.read()
    if frame is None:
      break
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.imshow('Video', frame)
    cv2.resizeWindow('Video', 800, 600)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    img = np.array(frame)

    out_img = img

    # request from the client side
    request = socket.recv_json(flags=0)
    
    if request.get('corr_id', False):
      result = {'corr_id': request['corr_id'],
        'shape': out_img.shape, 'dtype': str(out_img.dtype) }

      socket.send_json(result, flags = zmq.SNDMORE)
      socket.send(out_img, flags=0, copy=False, track=False)
    else:
      result={'corr_id': request['corr_id'], 'shape': None, 'dtype': None}
      socket.send_json(result)

if __name__ == '__main__':

  import sys

  if len(sys.argv) < 2:
    log.error('Missing arguments.')
    log.error('Usage: <image_file_name>')
    sys.exit(-1)

  IMAGE_FILE_NAME = sys.argv[1]
  main(SERVICE_SOCKET, IMAGE_FILE_NAME)
