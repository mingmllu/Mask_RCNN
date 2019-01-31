#!/usr/bin/env python


import zmq
import numpy as np
import uuid
import sys
import matplotlib.pyplot as plt
import time
from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils
import time
import cv2
import json
import argparse
from datetime import datetime


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--server-ip", help="Server host IP address", default='localhost')
parser.add_argument("--port", help="Port number associated with the socket", default=5566)
parser.add_argument("--record-video", help="File name of the recorded video", default='')
args = parser.parse_args()

server_ip = args.server_ip
port = args.port
video_file_name = args.record_video

SERVICE_SOCKET='tcp://' + server_ip + ':' + str(port)

ctx=zmq.Context()
socket=ctx.socket(zmq.REQ)

socket.connect(SERVICE_SOCKET)

if not video_file_name:
  # Define the codec and create VideoWriter object.The output is stored in the file.
  now = datetime.now().strftime("%Y%m%d%H%M%S")
  video_file_name = "video_{}_{}.avi".format(port, now)

frame_width = 640
frame_height = 360
video_sink = cv2.VideoWriter(video_file_name, 
            cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

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

    video_sink.write(cv2.resize(frame, (frame_width, frame_height)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()

socket.close()
ctx.term()
