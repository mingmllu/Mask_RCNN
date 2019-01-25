#!/usr/bin/env python


import zmq
import numpy as np
import uuid
import sys
import matplotlib.pyplot as plt
import time
import pylab
from imutils.video import FileVideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import json
 

SERVICE_SOCKET='tcp://localhost:5566'

ctx=zmq.Context()
socket=ctx.socket(zmq.REQ)

socket.connect(SERVICE_SOCKET)

print("[INFO] starting to looking at socket ...")

fps = FPS().start()

while True:
  # prepare a request for service
  corr_id = str(uuid.uuid4())
  request = {'corr_id': corr_id}

  socket.send_json(request)

  # wait for the reply from the service
  reply = socket.recv_multipart(flags=0)
  # the reply will be a list: the first item is metadata in raw bytes
  # the second item will be a video frame
  metadata_raw_bytes = reply[0]
  s = metadata_raw_bytes.decode("utf-8")
  data_double_quotes = s.replace("\'", "\"") #JSON strings must use double quotes
  metadata = json.loads(data_double_quotes)
  assert(metadata['corr_id'] == corr_id)

  if metadata.get('shape', None):
    video_frame_data = reply[1]
    buf = memoryview(video_frame_data)
    frame = np.frombuffer(buf, dtype = metadata['dtype']).reshape(metadata['shape'])

    cv2.namedWindow('Processed Video', cv2.WINDOW_NORMAL)
    cv2.imshow('Processed Video', frame)
    cv2.resizeWindow('Processed Video', 800,600)

    cv2.waitKey(1)
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
fvs.stop()

socket.close()
ctx.term()