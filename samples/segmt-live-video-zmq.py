#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.


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
    if os.getenv('IMAGE_MAX_DIM'):
       IMAGE_MAX_DIM = int(os.getenv('IMAGE_MAX_DIM'))
    if os.getenv('IMAGE_MIN_DIM'):
       IMAGE_MIN_DIM = int(os.getenv('IMAGE_MIN_DIM'))

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
import time

class MaskRCNNTracker():
  """Implements tracker based segmentation ouputs.

  Params:
  - 

  Inputs: 
  - 

  Output:
  A dictionay that maps the current frame's instance indexes to 
  the unique instance IDs that identify individual objects
  """

  def __init__(self):
    self.instance_id_manager = 0
    self.dict_instance_history = {}
    self.dict_trajectories = {}
    self.instance_memory_length = 2
    self.frame_number = 0  # the current frame number
    self.image_size = None # the image size (x, y) of the current frame
    self.dict_location_prediction = {}

  def fill_polygons_in_bounding_map(self, poly_vertices):
    """
    Given one or multiple ploygons rach consisting of a sequence of vertices, 
    determine a box or map that encloses them. Then fill the polygon(s) within
    the map and calculate its area.
    Input: 
    - poly_vertices: A list of polygons. Each item is a list of points [x,y]
    """
    left = 10000 # sufficiently large coordinate in x
    right = 0    # the minimum possible coordinate in x
    top = 10000  # sufficiently large coordinate in y
    bottom = 0   # the minimum possible coordinate in y
    # polyVertices: a list of N-by-2 arrays
    for poly in poly_vertices:
      left = min(left, np.amin(poly[:,0]))
      right = max(right, np.amax(poly[:,0]))
      top = min(top, np.amin(poly[:,1]))
      bottom = max(bottom, np.amax(poly[:,1]))
    pts = []
    for poly in poly_vertices:
      pts.append(poly-np.array([left,top]))
    # This map is a 2-D array
    map = np.zeros((bottom-top+1, right-left+1),dtype=np.uint8)
    # mask the area
    cv2.fillPoly(map, pts, color=(255))
    polyArea = np.count_nonzero(map)
    return (left, top, right, bottom, map, polyArea, self.frame_number)

  def compute_intersection_polygons(self, tuplePolygonA, tuplePolygonB):
    """
    Calculate intersection between two regions each outlined by one 
    or multiple polygons. 
    Inputs:
    - tuplePolygonA, tuplePolygonB: A tuple to represent a region outlined 
    by one or multiple polygons. See the output of method 
    "fill_polygons_in_bounding_map".
    Return: Intersection over Union (IoU) in the range from 0 to 1.0
    """
    # tuplePolygonA and tuplePolygonB
    # (xmin, ymin, xmax, ymax, filledPolygon2Dmap, frame_number)
    A_left = tuplePolygonA[0]
    A_right = tuplePolygonA[2]
    A_top = tuplePolygonA[1]
    A_bottom = tuplePolygonA[3]
    B_left = tuplePolygonB[0]
    B_right = tuplePolygonB[2]
    B_top = tuplePolygonB[1]
    B_bottom = tuplePolygonB[3]
    # check if the two maps intersect
    if B_left >= A_right or B_top >= A_bottom:
      return 0
    if A_left >= B_right or A_top >= B_bottom:
      return 0
    # calculate the overlapping part of the two bounding maps
    Overlap_left = max(A_left, B_left)
    Overlap_right = min(A_right, B_right)
    Overlap_top = max(A_top, B_top)
    Overlap_bottom = min(A_bottom, B_bottom)
    # get the overlapping part within the two maps respectively
    Overlap_A_map = tuplePolygonA[4][(Overlap_top-A_top):(min(A_bottom,Overlap_bottom)-A_top+1),
                    (Overlap_left-A_left):(min(A_right,Overlap_right)-A_left+1)]
    Overlap_B_map = tuplePolygonB[4][(Overlap_top-B_top):(min(B_bottom,Overlap_bottom)-B_top+1),
                    (Overlap_left-B_left):(min(B_right,Overlap_right)-B_left+1)]
    # calculate the intersection between the two silhouettes within the overlapping part
    Overlap_map_boolean = np.logical_and(Overlap_A_map, Overlap_B_map)
    # calculate the area of silhouette intersection
    Overlap_count = np.count_nonzero(Overlap_map_boolean)
    Union_count = tuplePolygonA[5] + tuplePolygonB[5] - Overlap_count
    return Overlap_count/Union_count

  def update_buffers(self):
    # Update the buffers (dictionaries) for the past detection results
    uid_list = list(self.dict_instance_history.keys())
    for uid in uid_list:
      if len(self.dict_instance_history[uid]) > self.instance_memory_length:
        self.dict_instance_history[uid].pop(0) # discard the oldest one
      while (len(self.dict_instance_history[uid]) > 0):
        if (self.frame_number - self.dict_instance_history[uid][0][6]) > 20: # discard stale frames
          self.dict_instance_history[uid].pop(0)
        else:
          break
    uid_list = list(self.dict_instance_history.keys())
    for uid in uid_list:
      if len(self.dict_instance_history[uid]) == 0:
        self.dict_instance_history.pop(uid)
        if uid in self.dict_trajectories:
          #self.save_trajectory_to_textfile(uid, "location")
          self.dict_trajectories.pop(uid)

    for uid in self.dict_trajectories:
      if (len(self.dict_trajectories[uid]) > 80):
        self.dict_trajectories[uid].pop(0)

  def receive_first_segmentation_output(self, results, class_names, image_size):
    """
    This method is called when the segmentation results for the very first frame received
    Input: 
    - results: segmentation results as output of Mask R-CNN 
    - class_names: list of class names of the dataset
    - image_size: image size in format (x, y)
    Output:
    - Tuple: 
      item 0: the current instance ID to assigned unique ID (dict)
      item 1: Contours for current instances (dict)
    """
    boxes = results['rois']
    masks = results['masks']
    class_ids = results['class_ids']

    self.image_size = image_size

    # Number of instances
    N = boxes.shape[0]
    if not N:
        return None
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
    # increment the frame counter
    self.frame_number = 1

    # Find the instances of interest, e.g., persons
    instances_of_interest = []
    for i in range(N):
      class_id = class_ids[i]
      if class_id == class_names.index('person'):
        instances_of_interest.append(i)

    # Find the contours that cover detected instances
    dict_contours = {}
    for i in instances_of_interest:
      # Mask
      mask = masks[:, :, i]
      # Mask Polygon
      # Pad to ensure proper polygons for masks that touch image edges.
      padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
      padded_mask[1:-1, 1:-1] = mask
      dict_contours[i] = find_contours(padded_mask, 0.5)
    
    # Analyze the contours and calculate the areas
    dict_polygons_in_bounding_map = {}
    for i in dict_contours:
      pts2d = []  # each element is an array of the shape (-1,2)
      for c in dict_contours[i]: # the value is a list
        pts2d.append(c.astype(np.int32))
      dict_polygons_in_bounding_map[i] = self.fill_polygons_in_bounding_map(pts2d)

    # Initialize the buffers
    dict_inst_index_to_uid = {} # mapping current frame's instance index to unique ID
    assert self.instance_id_manager == 0
    for i in dict_polygons_in_bounding_map:
      self.instance_id_manager += 1
      uid = self.instance_id_manager
      dict_inst_index_to_uid[i] = uid
      self.dict_instance_history[uid] = [dict_polygons_in_bounding_map[i]]
      y1, x1, y2, x2 = boxes[i]
      self.dict_trajectories[uid] = [[self.frame_number, (x1 + x2)//2, (y1 + y2)//2]]

    # calculate the center of the box that encloses a instance's contour
    dict_box_center = {}
    for i in dict_polygons_in_bounding_map:
      cy = (dict_polygons_in_bounding_map[i][0] + dict_polygons_in_bounding_map[i][2])//2
      cx = (dict_polygons_in_bounding_map[i][1] + dict_polygons_in_bounding_map[i][3])//2
      dict_box_center[i] = (cx, cy)

    # predict the locations of indentified instances in the next frame
    self.dict_location_prediction = {}
    for uid in self.dict_trajectories:
      self.dict_location_prediction[uid] = self.predict_location(uid)

    return (dict_inst_index_to_uid, dict_contours, dict_box_center)

  def receive_subsequent_segmentation_output(self, results, class_names, image_size):
    """
    Update tracker states upon new detection results
    Input: 
    - results: segmentation results as output of Mask R-CNN 
    - class_names: list of class names of the dataset
    - image_size: image size in format (x, y)
    Output:
    - Tuple: 
      item 0: the current instance ID to assigned unique ID (dict)
      item 1: Contours for current instances (dict)
    """
    boxes = results['rois']
    masks = results['masks']
    class_ids = results['class_ids']

    self.image_size = image_size

    # Number of instances
    N = boxes.shape[0]
    if not N:
        return None
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
    # increment the frame counter
    self.frame_number += 1
    
    # pop up the old data if necessary
    self.update_buffers()

    # Find the instances of interest, e.g., persons
    instances_of_interest = []
    for i in range(N):
      class_id = class_ids[i]
      if class_id == class_names.index('person'):
        instances_of_interest.append(i)

    # Find the contours that cover detected instances
    dict_contours = {}
    for i in instances_of_interest:
      # Mask
      mask = masks[:, :, i]
      # Mask Polygon
      # Pad to ensure proper polygons for masks that touch image edges.
      padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
      padded_mask[1:-1, 1:-1] = mask
      dict_contours[i] = find_contours(padded_mask, 0.5)
    
    # Analyze the contours and calculate the areas
    dict_polygons_in_bounding_map = {}
    for i in dict_contours:
      pts2d = []  # each element is an array of the shape (-1,2)
      for c in dict_contours[i]: # the value is a list
        pts2d.append(c.astype(np.int32))
      dict_polygons_in_bounding_map[i] = self.fill_polygons_in_bounding_map(pts2d)

    # Initialize the buffers for the past detection results
    if self.instance_id_manager == 0:
      for i in dict_polygons_in_bounding_map:
        self.instance_id_manager += 1
        uid = self.instance_id_manager
        self.dict_instance_history[uid] = [dict_polygons_in_bounding_map[i]]
        y1, x1, y2, x2 = boxes[i]
        self.dict_trajectories[uid] = [[self.frame_number, (x1 + x2)//2, (y1 + y2)//2]]

    # Coorespondence between existing instances and the instances in the current frame
    dict_inst_index_to_uid = {} # mapping current frame's instance index to unique ID
    list_matching_scores = []
    for i in dict_polygons_in_bounding_map:
      uid_matching = 0 # invalid ID
      max_iou = 0.0 # how much does it match the existing detected instances
      # here "uid" is a unique ID assigned to each detected instance
      for uid in self.dict_instance_history:
        for contour_map in reversed(self.dict_instance_history[uid]):
          iou = self.compute_intersection_polygons(dict_polygons_in_bounding_map[i], contour_map)
          if iou > max_iou:
            max_iou = iou
            uid_matching = uid
      if max_iou > 0:
        list_matching_scores.append((i, uid_matching, max_iou))
    list_matching_scores.sort(key=lambda item: item[2], reverse=True) # in decending order 
    uid_set = set(self.dict_instance_history.keys())
    for e in list_matching_scores: # e is a tuple
      i = e[0] # the instance ID in the current frame
      uid = e[1]  # unique existing instance ID
      iou_score = e[2]
      if iou_score > 0.25 and uid in uid_set:
        uid_set.remove(uid)  # this unique ID is claimed and won't be taken by other instances
        dict_inst_index_to_uid[i] = uid
        self.dict_instance_history[uid].append(dict_polygons_in_bounding_map[i]) # store the current frame

    # What if the instances do not match any of the existing identified instances ? 
    for i in dict_polygons_in_bounding_map:
      if i not in dict_inst_index_to_uid: # this would be a new instance
        self.instance_id_manager += 1
        uid = self.instance_id_manager
        self.dict_instance_history[uid] = [dict_polygons_in_bounding_map[i]]
        dict_inst_index_to_uid[i] = uid
    # calculate the center of the box that encloses a instance's contour
    dict_box_center = {}
    for i in dict_polygons_in_bounding_map:
      cy = (dict_polygons_in_bounding_map[i][0] + dict_polygons_in_bounding_map[i][2])//2
      cx = (dict_polygons_in_bounding_map[i][1] + dict_polygons_in_bounding_map[i][3])//2
      dict_box_center[i] = (cx, cy)
    
    for i in dict_inst_index_to_uid:
      y1, x1, y2, x2 = boxes[i]
      uid = dict_inst_index_to_uid[i]
      if uid not in self.dict_trajectories:
        self.dict_trajectories[uid] = [[self.frame_number, (x1 + x2)//2, (y1 + y2)//2]]
      else:
        self.dict_trajectories[uid].append([self.frame_number, (x1 + x2)//2, (y1 + y2)//2])

    # predict the locations of indentified instances in the next frame
    self.dict_location_prediction = {}
    for uid in self.dict_trajectories:
      self.dict_location_prediction[uid] = self.predict_location(uid)

    return (dict_inst_index_to_uid, dict_contours, dict_box_center)



  def receive_segmentation_output(self, results, class_names, image_size):
    """
    Update tracker states upon new detection results
    Input: 
    - results: segmentation results as output of Mask R-CNN 
    - class_names: list of class names of the dataset
    - image_size: image size in format (x, y)
    Output:
    - Tuple: 
      item 0: the current instance ID to assigned unique ID (dict)
      item 1: Contours for current instances (dict)
    """

    if self.instance_id_manager == 0:
      return self.receive_first_segmentation_output(results, class_names, image_size)
    else:
      return self.receive_subsequent_segmentation_output(results, class_names, image_size)


  def save_trajectory_to_textfile(self, uid, fname):
    """
    Dump a specified instance's location trajectory to a text file
    Input:
    - uid: Unique instance ID
    - fname: out filename
    """
    if uid in self.dict_trajectories:
      outfile = open(str(fname) + "_%04d"%(uid) + ".txt", "w")
      for u in self.dict_trajectories[uid]:
        outfile.write(str(u[0])+"\t"+str(u[1])+"\t"+str(u[2])+"\n")
      outfile.close()

  def estimate_velocity(self, uid):
    """
    Return estimated velocity
    """
    if uid not in self.dict_trajectories:
      return None
    pos = np.array(self.dict_trajectories[uid])
    m = pos.shape[0] # the number of points (memory for the past images)
    if m < 2: # single point
      return (0, 0)
    # partition the set of points
    x0, y0 = pos[0:m//2, 1].mean(), pos[0:m//2, 2].mean()
    x1, y1 = pos[m//2:, 1].mean(), pos[m//2:, 2].mean()
    timespan = np.amax([1.0, (pos[-1, 0] - pos[0, 0])/2])
    return (round((x1 - x0)/timespan), round((y1 - y0)/timespan))  # unit: pixels per frame
    
  def predict_location(self, uid):
    """
    Predict the location (x, y) of specified instance in the next frame
    """
    if uid not in self.dict_trajectories:
      return None
    _, x, y = self.dict_trajectories[uid][-1] # the latest (last) item
    v = self.estimate_velocity(uid)
    x_t = min([max([0, x + v[0]]), self.image_size[0]])
    y_t = min([max([0, y + v[1]]), self.image_size[1]])
    #print("uid", uid, "Velocity", v,"prediction: ","x", x, "->", x_t, "y", y, "->", y_t)
    return (x_t, y_t)
    



def generate_masked_image(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, tracking=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """

    # Number of instances
    N = boxes.shape[0]
    if not N:
        return image
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Find the instances of interest, e.g., persons
    instances_of_interest = []
    for i in range(N):
      class_id = class_ids[i]
      if class_id == class_names.index('person'):
        instances_of_interest.append(i)

    # Generate random colors
    diff_colors_person = False
    if not colors:
      diff_colors_person = True
    colors = colors or visualize.random_colors(N)

    # Determine the color of each detected instance
    dict_colors = {}
    if tracking:
      for i in tracking[0]:
        dict_colors[i] = colors[tracking[0][i]%len(colors)]

    masked_image = image.astype(np.uint32).copy()
    list_contours = []
    for i in instances_of_interest:
        class_id = class_ids[i]
        if diff_colors_person:
          color = colors[i%len(colors)]
        else:
          color = colors[class_id%len(colors)]

        if i in dict_colors:
          color = dict_colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        #y1, x1, y2, x2 = boxes[i]
        #if show_bbox:
        #    p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
        #                        alpha=0.7, linestyle="dashed",
        #                        edgecolor=color, facecolor='none')
        #    ax.add_patch(p)

        # Label
        #if not captions:
        #    class_id = class_ids[i]
        #    score = scores[i] if scores is not None else None
        #    label = class_names[class_id]
        #    x = random.randint(x1, (x1 + x2) // 2)
        #    caption = "{} {:.3f}".format(label, score) if score else label
        #else:
        #    caption = captions[i]
        #ax.text(x1, y1 + 8, caption,
        #        color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

    masked_image_uint8 = masked_image.astype(np.uint8)
    
    dict_inst_index_to_uid = {}
    dict_contours = {}
    dict_box_center = {}
    if tracking:
      dict_inst_index_to_uid = tracking[0]
      dict_contours = tracking[1]
      dict_box_center = tracking[2]

    for i in dict_contours:
      contours = dict_contours[i]
      # contours is a list
      # cv2.polylines requires shape (-1,1,2)
      pts3d = []
      for c in contours:
        # switch x with y otherwise the contours will be rotated by 90 degrees
        pts3d.append(c.astype(np.int32).reshape((-1, 1, 2))[:,:,[1,0]])
      cv2.polylines(masked_image_uint8, pts3d, True, (0, 255, 255))
      uid = dict_inst_index_to_uid[i]
      center = dict_box_center[i]
      font = cv2.FONT_HERSHEY_SIMPLEX
      fontScale = 0.5
      fontColor = (0,255,255)
      lineType = 1
      cv2.putText(masked_image_uint8, str(uid), center, font, fontScale, fontColor, lineType)
    return masked_image_uint8

import cv2
import numpy as np
import time

# Create a VideoCapture object
#image_source = 'http://108.53.114.166/mjpg/video.mjpg' # Newark overpass
#image_source = 'http://root:fitecam@135.222.247.179:9122/mjpg/video.mjpg' # Kiosk
image_source = 'http://anomaly:lucent@135.104.127.10:58117/mjpg/video.mjpg' # Cafe
if os.getenv('IMAGE_SOURCE'):
  image_source = os.getenv('IMAGE_SOURCE')

def open_source_video(image_source):
  # allow multiple attempts to open video source
  max_num_attempts = 10
  count_attempts = 1
  cap = cv2.VideoCapture(image_source)
  # Check if camera opened successfully
  while (cap.isOpened() == False):
    print("Unable to open image source %s: %d out of %d"%(image_source, count_attempts, max_num_attempts))
    if count_attempts == max_num_attempts:
      break
    time.sleep(0.5)
    count_attempts += 1
    cap = cv2.VideoCapture(image_source)
  return cap # return a video capture object that is in open state 


def create_video_writer(cap, filename):
  # Default resolutions of the frame are obtained.The default resolutions are system dependent.
  # We convert the resolutions from float to integer.
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
 
  # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
  outputfilename = os.path.join(VIDEO_OUTPUT_DIR, filename + '.avi')
  out = cv2.VideoWriter(outputfilename, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
  return out

if os.getenv('RANDOM_MASK_COLORS'):
  colors = None  # random colors from frame to frame
else:
  colors = visualize.random_colors(20) # assume that there are 20 instances

import os
import zmq

SOURCE_IMAGE_RESIZE_FACTOR = None
if os.getenv('SOURCE_IMAGE_RESIZE_FACTOR'):
  SOURCE_IMAGE_RESIZE_FACTOR = float(os.getenv('SOURCE_IMAGE_RESIZE_FACTOR'))

def detect_and_save_frames(cap, model, max_frames_to_be_saved, video_sink):
  # A counter for frames that have been written to the output file so far
  n_frames = 0

  tracker = MaskRCNNTracker()

  while(True):
    ret, frame = cap.read()
    if ret == False: 
      break

    # reduce the input image size to speed up the masking of the image
    if SOURCE_IMAGE_RESIZE_FACTOR and SOURCE_IMAGE_RESIZE_FACTOR < 1:
      fw = fh = SOURCE_IMAGE_RESIZE_FACTOR
      frame = cv2.resize(frame, (0,0), fx=fw, fy=fh)

    start_time = time.time()
    results = model.detect([frame], verbose=0)
    finish_time = time.time()
    print("Elapsed time per frame = %f"%(finish_time - start_time))
    r = results[0]
    image_size = (frame.shape[1], frame.shape[0])
    tracking_predictions = tracker.receive_segmentation_output(r, class_names, image_size)
    masked_frame = generate_masked_image(frame, r['rois'], r['masks'], r['class_ids'], 
                   class_names, r['scores'], colors=colors, tracking=tracking_predictions)
    print("Rendering %f"%(time.time() - finish_time))
 
    # Write the frame into the file 'output.avi'
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    video_sink.write(cv2.resize(masked_frame, (frame_width, frame_height)))
    n_frames += 1

    skimage.io.imsave(os.path.join(VIDEO_OUTPUT_DIR, 'masked_frame_%05d.jpg'%(n_frames)), masked_frame)

    print("Frame %d out of %d saved " % (n_frames, max_frames_to_be_saved))
    if n_frames == max_frames_to_be_saved:
      break

def detect_and_send_frames(cap, model, socket):
  # A counter for frames that have been written to the output file so far
  n_frames = 0

  tracker = MaskRCNNTracker()

  while(True):
    ret, frame = cap.read()
    if ret == False:
      break
    
    # reduce the input image size to speed up the masking of the image
    if SOURCE_IMAGE_RESIZE_FACTOR and SOURCE_IMAGE_RESIZE_FACTOR < 1:
      fw = fh = SOURCE_IMAGE_RESIZE_FACTOR
      frame = cv2.resize(frame, (0,0), fx=fw, fy=fh)

    start_time = time.time()
    results = model.detect([frame], verbose=0)
    finish_time = time.time()
    print("Elapsed time per frame = %f"%(finish_time - start_time))
    r = results[0]
    image_size = (frame.shape[1], frame.shape[0])
    tracking_predictions = tracker.receive_segmentation_output(r, class_names, image_size)
    masked_frame = generate_masked_image(frame, r['rois'], r['masks'], r['class_ids'], 
                   class_names, r['scores'], colors=colors, tracking=tracking_predictions)
    print("Rendering %f"%(time.time() - finish_time))
    n_frames += 1
    print("Frame %d" % (n_frames))

    # listening on request from the client side
    request = socket.recv_json(flags=0)
    
    if request.get('corr_id', False):
      result = {'corr_id': request['corr_id'],
        'shape': masked_frame.shape, 'dtype': str(masked_frame.dtype) }
      socket.send_json(result, flags = zmq.SNDMORE)
      socket.send(masked_frame, flags=0, copy=False, track=False)
    else:
      result={'corr_id': request['corr_id'], 'shape': None, 'dtype': None}
      socket.send_json(result)




SERVICE_PORT = os.getenv('SKT_PORT', None)
if SERVICE_PORT:
  SERVICE_SOCKET = "tcp://*:%s"%(SERVICE_PORT)
  context = zmq.Context()
  socket = context.socket(zmq.REP)
  # ZMQ server must be listening on request first
  socket.bind(SERVICE_SOCKET)
  print("Listening on request from client side ...")
  request = socket.recv_json(flags=0)
  result={'corr_id': request['corr_id'], 'shape': None, 'dtype': None}
  socket.send_json(result)
  # now open video to avoid possible ffmpeg overread error
  cap = open_source_video(image_source)
  if (cap.isOpened() == False):
    exit()
  detect_and_send_frames(cap, model, socket)
else:
  cap = open_source_video(image_source)
  if (cap.isOpened() == False):
    exit()
  output_filename = os.getenv('OUTPUT_VIDEO_FILENAME')
  if not output_filename:
    output_filename = 'video_segmentation_mjpg4'
  out = create_video_writer(cap, output_filename)
  # The maximum number of frames to be written
  max_number_frames_to_be_saved = os.getenv('MAX_FRAMES_TO_BE_SAVED')
  if not max_number_frames_to_be_saved:
    max_number_frames_to_be_saved = 100
  detect_and_save_frames(cap, model, int(max_number_frames_to_be_saved), out)
  out.release()

# When everything done, release the video capture
cap.release()


