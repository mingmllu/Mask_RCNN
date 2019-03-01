
import os
import sys
import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from skimage.measure import find_contours


class MaskRCNNTracker():
  """Implements tracker based on segmentation outputs.

  Params:
  - 

  Inputs: 
  - 

  Output:
  A dictionay that maps the current frame's instance indexes to 
  the unique instance IDs that identify individual objects
  """

  def __init__(self, class_names):
    """
    class_names: list of class names of the dataset. 
                 used to map detected instances to classes
    """
    self.class_names = class_names
    self.instance_memory_length = 2
    self.image_size = None # the image size (x, y) of the current frame
    self.occlusion_factor_thresh = 0.4 # parameter
    # the inner area conssist of the inner grids not touching any sides
    self.N_divide_width = 8  # the number of grids along x
    self.N_divide_height = 4 # the number of grids along y
    self.left_top_right_bottom = None # A rectangle for inner frame 
    # once the number of consecutive inactivity frames exceeds the period,
    # reset the tracker
    self.max_inactivity_period = 50 # in frames
    self.frame_stale_timer = 20 # do not keep info about a frame that is too old
    # the maximum Euclidean histogram distance for two similar histograms
    self.hist_dissimilarity_thresh = 0.2
    self.reset()

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

  def reset(self):
    """
    Reset the tracker: flush all buffers and reset all internal dynamic state variables 
    """
    self.inactivity_counter = 0 # the number of consectutive frames where no instances detected 
    self.instance_id_manager = 0
    self.dict_instance_history = {}
    self.dict_trajectories = {}
    self.frame_number = 0  # the current frame number
    self.dict_location_prediction = {}
    self.dict_appearance_prediction = {}
    # store each instance's states. 
    # For example, "occlusion" 
    self.dict_instance_states = {}
    self.dict_hue_histogram = {} # keys: ID assigned to instance under track
    # If an instance is deemed to be out of track, its relevant information like color
    # histogram will be stored in the dictionary. It may be re-claimed later based on
    # color matching.
    self.dict_instances_out_of_track = {} # keys: instance unique ID

  def update_buffers(self):
    # Update the buffers (dictionaries) for the past detection results
    uid_list = list(self.dict_instance_history.keys())
    for uid in uid_list:
      if len(self.dict_instance_history[uid]) > self.instance_memory_length:
        self.dict_instance_history[uid].pop(0) # discard the oldest one
      while (len(self.dict_instance_history[uid]) > 0):
        if (self.frame_number - self.dict_instance_history[uid][0][6]) > self.frame_stale_timer:
          self.dict_instance_history[uid].pop(0)
        else:
          break
    uid_list = list(self.dict_instance_history.keys())
    for uid in uid_list:
      if len(self.dict_instance_history[uid]) == 0:
        self.dict_instance_history.pop(uid)
        self.dict_instances_out_of_track[uid] = {}  # keep it in case it will be re-claimed
        if uid in self.dict_trajectories:
          #self.save_trajectory_to_textfile(uid, "location")
          self.dict_trajectories.pop(uid)
        if uid in self.dict_location_prediction:
          self.dict_location_prediction.pop(uid)
        if uid in self.dict_instance_states:
          self.dict_instance_states.pop(uid)    
        if uid in self.dict_appearance_prediction:
          self.dict_appearance_prediction.pop(uid)

    for uid in self.dict_trajectories:
      if (len(self.dict_trajectories[uid]) > 10):
        self.dict_trajectories[uid].pop(0)

    # Remove the records for instances that have already disappeared forever
    uid_list = list(self.dict_hue_histogram.keys())
    for uid in uid_list:
      if not uid in self.dict_trajectories:
        self.dict_instances_out_of_track[uid]['hist_hue'] = self.get_average_histogram_hue(uid)
        self.dict_instances_out_of_track[uid]['frame_number'] = self.frame_number
        self.dict_hue_histogram.pop(uid)
    for uid in self.dict_hue_histogram:
      if len(self.dict_hue_histogram[uid]) > 4:
        self.dict_hue_histogram[uid].pop(0) # only keep the most recent histograms

    # Delete the instances that have been out of the scene for a long time
    uid_list = list(self.dict_instances_out_of_track.keys())
    for uid in uid_list:
      if self.frame_number - self.dict_instances_out_of_track[uid]['frame_number'] > 120:
        self.dict_instances_out_of_track.pop(uid)


  def receive_first_segmentation_output(self, results, image):
    """
    This method is called when the segmentation results for the very first frame received
    Input: 
    - results: segmentation results as output of Mask R-CNN 
    - image: the current image or video frame
    Output:
    - Tuple: 
      item 0: the current instance ID to assigned unique ID (dict)
      item 1: Contours for current instances (dict)
    """
    boxes = results['rois']
    masks = results['masks']
    class_ids = results['class_ids']
    scores = results['scores']


    # Number of instances
    N = boxes.shape[0]
    if not N:
        self.inactivity_counter += 1
        return None
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
    # increment the frame counter
    self.frame_number = 1

    # Find the instances of interest, e.g., persons
    instances_of_interest = []
    for i in range(N):
      class_id = class_ids[i]
      if class_id == self.class_names.index('person') and scores[i] >= 0.75:
        instances_of_interest.append(i)

    if len(instances_of_interest) == 0:
      self.inactivity_counter += 1
    else:
      self.inactivity_counter = 0

    # calculate the histograms of color (hue) for each segmented instances
    dict_histograms_hue = self.calculate_hue_histograms(instances_of_interest, masks, image)

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
      self.dict_hue_histogram[uid] = [dict_histograms_hue[i]]

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
      dx, dy = self.dict_location_prediction[uid][2:4]
      self.dict_appearance_prediction[uid] = self.shift_instance_appearance(uid, dx, dy)

    return (dict_inst_index_to_uid, dict_contours, dict_box_center)

  def receive_subsequent_segmentation_output(self, results, image):
    """
    Update tracker states upon new detection results
    Input: 
    - results: segmentation results as output of Mask R-CNN 
    - image: the current image or video frame
    Output:
    - Tuple: 
      item 0: the current instance ID to assigned unique ID (dict)
      item 1: Contours for current instances (dict)
    """
    boxes = results['rois']
    masks = results['masks']
    class_ids = results['class_ids']
    scores = results['scores']


    # Number of instances
    N = boxes.shape[0]
    if not N:
        self.inactivity_counter += 1
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
      if class_id == self.class_names.index('person') and scores[i] >= 0.75:
        instances_of_interest.append(i)

    if len(instances_of_interest) == 0:
      self.inactivity_counter += 1
      if self.inactivity_counter >= self.max_inactivity_period:
        self.reset()
    else:
      self.inactivity_counter = 0

    # calculate the histograms of color (hue) for each segmented instances
    dict_histograms_hue = self.calculate_hue_histograms(instances_of_interest, masks, image)

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

    # Correspondence between existing instances and the instances in the current frame
    dict_inst_index_to_uid = {} # mapping current frame's instance index to unique ID
    list_matching_scores = []
    dict_inst_occlusion = {}
    for i in dict_polygons_in_bounding_map:
      uid_matching = 0 # invalid ID
      max_iou = 0.0 # how much does it match the existing detected instances
      # here "uid" is a unique ID assigned to each detected instance
      for uid in self.dict_instance_history:
        contour_map = self.get_instance_appearance(uid)
        iou = self.compute_intersection_polygons(dict_polygons_in_bounding_map[i], contour_map)
        if iou > max_iou:
          max_iou = iou
          uid_matching = uid
      if max_iou > 0:
        list_matching_scores.append((i, uid_matching, max_iou))
    list_matching_scores.sort(key=lambda item: item[2], reverse=True) # in decending order 
    uid_set = set(self.dict_instance_history.keys())
    # key = instance ID in the current frame
    # Values of IoU scores used for debugging purpose
    dict_instance_score_color_mismatch = {} 
    for e in list_matching_scores: # e is a tuple
      i = e[0] # the instance ID in the current frame
      uid = e[1]  # unique existing instance ID
      iou_score = e[2]
      if iou_score > 0.05 and uid in uid_set:
        if not self.is_occluded_next_frame(uid):
          hue_dissimilarity = self.calculate_distance_between_histograms(dict_histograms_hue[i], 
                                   self.dict_hue_histogram[uid][-1])
          if hue_dissimilarity < self.hist_dissimilarity_thresh:
            uid_set.remove(uid)  # this unique ID is claimed and won't be taken by other instances
            dict_inst_index_to_uid[i] = uid
            self.dict_instance_history[uid].append(dict_polygons_in_bounding_map[i]) # store the current frame
          else:
            dict_instance_score_color_mismatch[i] = iou_score # mismatch probably due to color contamination
        else:
          dict_inst_occlusion[i] = True

    # What if the instances do not match any of the existing identified instances ? 
    # The instances that appear suddenly within the inner area of frame may be false positives
    id_list = list(dict_polygons_in_bounding_map.keys())
    for i in id_list:
      if i not in dict_inst_index_to_uid:
        y1, x1, y2, x2 = boxes[i]
        # possibly a false positive or in occlusion
        is_within_inner_area = self.is_within_inner_area((x1 + x2)//2, (y1 + y2)//2)
        is_occluded = i in dict_inst_occlusion
        is_color_mismatch = i in dict_instance_score_color_mismatch # this may not be a new instance
        if is_within_inner_area or is_occluded or is_color_mismatch:
          dict_polygons_in_bounding_map.pop(i)
          dict_contours.pop(i)
          instances_of_interest.remove(i)
    # Reclaim the instances out of track if possible
    for i in dict_polygons_in_bounding_map:
      if i not in dict_inst_index_to_uid: # make sure it's not associated with any instance on track
        min_hue_dissimilarity = 1.0
        uid_min_hue_dissimilarity = 0
        for uid in self.dict_instances_out_of_track:
          histogram_color = self.dict_instances_out_of_track[uid]['hist_hue']
          hue_dissimilarity = self.calculate_distance_between_histograms(histogram_color, dict_histograms_hue[i])
          if hue_dissimilarity < min_hue_dissimilarity:
            min_hue_dissimilarity = hue_dissimilarity
            uid_min_hue_dissimilarity = uid
        if min_hue_dissimilarity < self.hist_dissimilarity_thresh:
          self.dict_instance_history[uid_min_hue_dissimilarity] = [dict_polygons_in_bounding_map[i]]
          dict_inst_index_to_uid[i] = uid_min_hue_dissimilarity
          self.dict_instances_out_of_track.pop(uid_min_hue_dissimilarity)
    # Now assign unique IDs to new instances
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

    # predict the locations of the identified instances in the next frame
    for uid in self.dict_trajectories:
      self.dict_location_prediction[uid] = self.predict_location(uid)
      dx, dy = self.dict_location_prediction[uid][2:4]
      self.dict_appearance_prediction[uid] = self.shift_instance_appearance(uid, dx, dy)

    list_occlusion = self.predict_occlusion(self.occlusion_factor_thresh)
    self.dict_instance_states = {}
    for uid in list_occlusion:
      self.dict_instance_states[uid] = dict(occlusion=True)

    for i in dict_inst_index_to_uid:
      uid = dict_inst_index_to_uid[i]
      if uid in self.dict_hue_histogram:
        self.dict_hue_histogram[uid].append(dict_histograms_hue[i])
      else:
        self.dict_hue_histogram[uid] = [dict_histograms_hue[i]]      

    return (dict_inst_index_to_uid, dict_contours, dict_box_center)



  def receive_segmentation_output(self, results, image):
    """
    Update tracker states upon new detection results
    Input: 
    - results: segmentation results as output of Mask R-CNN 
    - image: the current image or video frame
    Output:
    - Tuple: 
      item 0: the current instance ID to assigned unique ID (dict)
      item 1: Contours for current instances (dict)
    """
    self.image_size = (image.shape[1], image.shape[0])
    self.update_inner_frame_area()
    if self.instance_id_manager == 0:
      return self.receive_first_segmentation_output(results, image)
    else:
      return self.receive_subsequent_segmentation_output(results, image)


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

    assert uid in self.dict_instance_history
    dx = dy = 0  # displacement relative to the last true location
    if self.dict_instance_history[uid][-1][6] == self.frame_number:
      _, x, y = self.dict_trajectories[uid][-1] # the latest (last) item
    else:
      assert uid in self.dict_location_prediction
      x, y, dx, dy = self.dict_location_prediction[uid] # based on the prediction for the last frame
    v = self.estimate_velocity(uid)
    x_t = min([max([0, x + v[0]]), self.image_size[0]])
    y_t = min([max([0, y + v[1]]), self.image_size[1]])
    dx += v[0]  # accumulated displacement in x taking into account frames where trajectory not updated
    dy += v[1]  # accumulated displacement in y taking into account frames where trajectory not updated
  
    #print("uid", uid, "Velocity", v,"prediction: ","x", x, "->", x_t, "y", y, "->", y_t, "dx", dx, "dy", dy)
    return (x_t, y_t, dx, dy)
    
  def shift_instance_appearance(self, uid, dx, dy):
    """
    Generate the instance appearance at the predicted location (xpos, ypos) for the next frame.
    It is just a shift of the last appearance
    Inputs:
    - uid Unique ID of instance
    - dx, dy: location displacement relative to the last trajectory update
    """
    if uid not in self.dict_instance_history:
      return None
    last_profile = self.dict_instance_history[uid][-1]
    # sigh! because polygons rotated by 90 degrees, we have to switch x and y
    dy, dx = int(dx), int(dy)
    left = min([max([0, last_profile[0] + dx]), self.image_size[1]])
    right = min([max([0, last_profile[2] + dx]), self.image_size[1]])
    top = min([max([0, last_profile[1] + dy]), self.image_size[0]])
    bottom = min([max([0, last_profile[3] + dy]), self.image_size[0]])

    return (left, top, right, bottom, last_profile[4], last_profile[5], self.frame_number)

  def get_instance_appearance(self, uid):
    """
    Return instance's last appearance
    """
   
    if uid in self.dict_appearance_prediction:
      return self.dict_appearance_prediction[uid]
    elif uid in self.dict_instance_history:
      return self.dict_instance_history[uid][-1]
    else:
      return None

  def compute_occlusion_factor(self, tuplePolygonA, tuplePolygonB):
    """
    Calculate the occlusion factor of a region which may be partially
    or totally occluded by another region
    Inputs:
    - tuplePolygonA, tuplePolygonB: A tuple to represent a region outlined 
    by one or multiple polygons. See the output of method 
    "fill_polygons_in_bounding_map".
    Return: How much region A is occcluded by region B in the range from 0 to 1.0
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
    assert tuplePolygonA[5] > 0
    return Overlap_count/tuplePolygonA[5]

  def predict_occlusion(self, occlusion_factor_thresh):
    """
    Based on the predicted instance appearances for the next frame, find the existing 
    instances that will be partially or totally occluded  in the next frame
    Output: A list of instances to be occluded
    """
    output_list = []
    uid_list = list(self.dict_appearance_prediction.keys())
    num = len(uid_list)
    for uid1 in uid_list:
      of_max = 0
      for uid2 in uid_list:
        if uid1 == uid2:
          continue
        of = self.compute_occlusion_factor(self.dict_appearance_prediction[uid1], self.dict_appearance_prediction[uid2])
        if of > of_max:
          of_max = of
      if of_max > occlusion_factor_thresh:
        output_list.append(uid1)

    return output_list 

  def is_occluded_next_frame(self, uid):
    """
    Will the object be occluded in the next frame ?
    """
    if uid in self.dict_instance_states:
      if self.dict_instance_states[uid]['occlusion']:
        return True
    return False

  def update_inner_frame_area(self):
    """
    Given the frame size (i.e., self.image_size), determine the inner area of the frame
    """
    left = self.image_size[0]//self.N_divide_width
    right = (self.N_divide_width - 1) * self.image_size[0]//self.N_divide_width
    top = self.image_size[1]//self.N_divide_height
    bottom = (self.N_divide_height - 1) * self.image_size[1]//self.N_divide_height
    self.left_top_right_bottom = (left, top, right, bottom)

  def is_within_inner_area(self, x, y, leftside=True, rightside=True,topside=False, bottomside=True):
    """
    Given the location of an object, check if it is in the inner of the frame
    Inputs:
    - x, y: coordinates of location
    - leftside, rightside, topside, bottomside: if False, objects never get into or out of the frame 
    the frame across the side
    """
    if leftside and x < self.left_top_right_bottom[0]:
      return False
    if rightside and x > self.left_top_right_bottom[2]:
      return False
    if topside and y < self.left_top_right_bottom[1]:
      return False
    if bottomside and y > self.left_top_right_bottom[3]:
      return False
    
    return True

  def calculate_hue_histograms(self, instance_ids, masks, image):
    """  
    Calculate the histogram of hue for each segmented instance
    instance_ids: a list of instances of interest
    masks: 3D-array 
    image: video frame
    """

    dict_hue_histogram = {}
    num_bins = 36
    hue_range = [0,180] # for opencv
    for i in instance_ids:
      mask = masks[:, :, i]
      contour_indexes = np.where(mask == 1)
      b = image[:,:,0]
      g = image[:,:,1]
      r = image[:,:,2]
      b = b[contour_indexes].reshape(-1,1)
      g = g[contour_indexes].reshape(-1,1)
      r = r[contour_indexes].reshape(-1,1)
      bgr = np.stack((b, g, r), axis=-1)
      hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
      hist, bins = np.histogram(hsv[:,:,0].ravel(),bins=num_bins, range=hue_range, density=True)
      dict_hue_histogram[i] = hist * ((hue_range[1] - hue_range[0]) / num_bins)
    return dict_hue_histogram
  
  def calculate_distance_between_histograms(self, hist1, hist2):
    """
    Calculate the distance between two normalized histograms
    """
    assert hist1.shape == hist2.shape
    return np.linalg.norm(hist1 - hist2)

  def get_average_histogram_hue(self, uid):
    """
    Calcualte and return an instance's average histogram of hue
    """
    if uid not in self.dict_hue_histogram:
      return None
    avg_hist = np.zeros(self.dict_hue_histogram[uid][0].shape)
    for hist in self.dict_hue_histogram[uid]:
      avg_hist = avg_hist + hist
    return avg_hist / np.sum(avg_hist)  # normalized such that the sum is 1