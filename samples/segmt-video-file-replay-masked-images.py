#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:


import os
import keras

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
if 'PYTHONPATH' in os.environ:
    print("Please unset the environment variable PYTHONPATH if you got errors with pycocotools!")
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Directory of videos to be saved as detection results
VIDEO_OUTPUT_DIR = os.path.join(ROOT_DIR, "videos")

# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# ## Run Object Detection

from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import cv2

def generate_masked_image(image, boxes, masks, class_ids, class_names,
                      scores=None,
                      show_mask=True, show_bbox=True, show_score=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    show_mask, show_bbox: To show masks and bounding boxes or not
    show_score: To show scores or not
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        return image
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        class_id = class_ids[i]
        if class_id != class_names.index('person'):
            continue
        #color = colors[i]
        color = colors[class_id%len(colors)]
        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

    annotated_masked_image = masked_image.astype(np.uint8)
    white = (255,255,255)
    green = (0, 255, 0)
    for i in range(N):
        class_id = class_ids[i]
        if class_id != class_names.index('person'):
            continue
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            cv2.rectangle(annotated_masked_image, (x1, y1), (x2, y2), green, 1)
        # Label
        if not show_score:
            continue
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontColor = white
        lineType = 1
        cv2.putText(annotated_masked_image, caption, (x1, y1 + 8), font, fontScale, fontColor, lineType)
    return annotated_masked_image

import cv2
import numpy as np
import time

# Create a VideoCapture object
cap = cv2.VideoCapture('demo.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
  exit()

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
outputfilename = os.path.join(VIDEO_OUTPUT_DIR, 'video_segmentation_mjpg4.avi')
out = cv2.VideoWriter(outputfilename, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
# A counter for frames that have been written to the output file so far
n_frames = 0
# The maximum number of frames to be written
max_number_framed_to_be_saved = 250

colors = visualize.random_colors(10) # assume that there are 10 instances

SOURCE_IMAGE_RESIZE_FACTOR = None
if os.getenv('SOURCE_IMAGE_RESIZE_FACTOR'):
  SOURCE_IMAGE_RESIZE_FACTOR = float(os.getenv('SOURCE_IMAGE_RESIZE_FACTOR'))

while(True):
  ret, frame = cap.read()

  if ret == True: 

    # reduce the input image size to speed up the masking of the image
    if SOURCE_IMAGE_RESIZE_FACTOR and SOURCE_IMAGE_RESIZE_FACTOR < 1:
      fw = fh = SOURCE_IMAGE_RESIZE_FACTOR
      frame = cv2.resize(frame, (0,0), fx=fw, fy=fh)

    start_time = time.time()
    results = model.detect([frame], verbose=0)
    finish_time = time.time()
    print("Elapsed time per frame = %f"%(finish_time - start_time))
    r = results[0]
    masked_frame = generate_masked_image(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], 
                   colors=colors,
                   show_mask=True, show_bbox=True, show_score=True)
    print("Rendering %f"%(time.time() - finish_time))
    #skimage.io.imsave(os.path.join(VIDEO_OUTPUT_DIR, 'masked_frame_%05d.jpg'%(n_frames)), masked_frame)

    # Write the frame into the file 'output.avi'
    out.write(masked_frame)
    n_frames += 1
    print("Frame %d out of %d saved " % (n_frames, max_number_framed_to_be_saved))
    if n_frames == max_number_framed_to_be_saved:
      break
    
    # Display the resulting frame    
    cv2.imshow('Display live video while recording ... Type q to quit',masked_frame)
 
    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break 

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows() 







#%%
