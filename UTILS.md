## Notes about the file "utils.py"

Common utility functions and classes.

### Local variables
Variables                   | Definitions  | Comments
--------------------------- | ------------ | ----------------------------------------------------------------------------------------------
COCO_MODEL_URL | "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5" | URL from which to download the latest COCO trained weights

### Common utility functions

Funtions                    |  Arguments     | Return  | Comments
--------------------------- | -------------- | ------- | ----------------------------------------------------------------------------------
extract_bboxes(mask) | mask: [height, width, num_instances] | bbox array [num_instances, (y1, x1, y2, x2)] | Given the mask, find the bbox. In each axis (i.e., the x or y axis), find the firt and the last True pixel and then form the bounding box
compute_iou(box, boxes, box_area, boxes_area) | box: 1D vector [y1, x1, y2, x2] boxes: [boxes_count, (y1, x1, y2, x2)] box_area: float. the area of 'box' boxes_area: array of length boxes_count| IoU | Calculates IoU of the given box with the array of the given boxes
compute_overlaps(boxes1, boxes2) | boxes1, boxes2: [N, (y1, x1, y2, x2)] | Overlaps | Computes IoU overlaps between two sets of boxes
compute_overlaps_masks(masks1, masks2) | masks1, masks2: [Height, Width, instances] | IoU | Computes IoU overlaps between two sets of masks
non_max_suppression(boxes, scores, threshold) | indices of kept boxes |indices of kept boxes | Performs non-maximum suppression
apply_box_deltas(boxes, deltas) | boxes: [N, (y1, x1, y2, x2)].  deltas: [N, (dy, dx, log(dh), log(dw))] | | Applies the given deltas to the given boxes
box_refinement_graph(box, gt_box) | | | Compute refinement needed to transform box to gt_box (TF graph, symbolic ?)
box_refinement(box, gt_box) | | stacked [dy, dx, dh, dw] | Compute refinement needed to transform box to gt_box (numpy)

### Class Dataset(object)

To use it, create a new class that adds functions specific to the dataset you want to use. 

Attribute                  | Type                    | Comments
-------------------------- | ----------------------- | ------------------------------------------------------------------------------------
class_info | List | A list of class information. Each item follows JSON format. The first item is corresponging to background: {"source": "", "id": 0, "name": "BG"}
_image_ids  | numpy.ndarray | np.arange(self.num_images)
image_info  | List | Each item is a dictionary.
source_class_ids | dict ?? | Map sources to class_ids they support. The values are lists.
num_classes | Integer | Number of classes. Computed property when the method "prepare" is called
class_ids | numpy.ndarray | Evenly spaced intergers staring from 0. Computed property when the method "prepare" is called.
class_names | List | A list of class names
num_images | Interger | len(self.image_info)
class_from_source_map | dict ?? | Computed property when the method "prepare" is called. Map "source.id" to local id. The local IDs are sequential without gap. The source IDs may have gaps.
image_from_source_map | dict ?? | Computed property when the method "prepare" is called. Map "source.id" to local id. The local IDs are sequential without gap. The source IDs may have gaps.
sources  | List | A list of class sources. Computed property when the method "prepare" is called.



Method                     | Return                  | Comments
-------------------------- | ----------------------- | ------------------------------------------------------------------------------------
add_class(self, source, class_id, class_name) | | Add a new class to the list "class_info". Each item follow JSON format: {"source": source,"id": class_id,"name": class_name}. *Source name cannot contain a dot*
add_image(self, source, image_id, path, **kwargs) | | Add an item to the list "image_info": {"id": image_id, "source": source, "path": path, }. The keyword argument will be merged into the item if specified.
image_reference(self, image_id) | string | Return a link to the image in its source Website or details about the image that help looking it up or debugging it. Override for your dataset, but pass to this function if you encounter images not in your dataset
prepare(self, class_map=None) | | Prepares the Dataset class for use
map_source_class_id(self, source_class_id) | Local class ID | Takes a source class ID and returns the int class ID assigned to it. dataset.map_source_class_id("coco.12") -> 23
get_source_class_id(self, class_id, source) | Source class ID | Map an internal class ID to the corresponding class ID in the source dataset.
image_ids(self) | self._image_ids | Decorated by the method Property
def source_image_link(self, image_id) | Path or URL | "image_id" is the one locally assigned ???. Returns the path or URL to the image. Override this to return a URL to the image if it's available online for easy debugging.
load_image(self, image_id) | numpy.ndarray | Using "skimage.io.imread" to load the specified image from its URL or path and return a [H,W,3] Numpy array. For grayscle images, converted to RGB for consistency. If image has an alpha channel, remove it for consistency
load_mask(self, image_id) | tuple (masks, class_ids) | Load instance masks from source dataset for the given image. Different datasets use different ways to store masks. Override this method to load instance masks and return them in the form of an array of binary masks of shape [height, width, instances]. masks: A bool array of shape [height, width, instance count] with a binary mask per instanceclass_ids: a 1D array of class IDs of the instance masks

### Functions
Function                     | Return                  | Comments
-------------------------- | ----------------------- | ------------------------------------------------------------------------------------
resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square") | tuple: the resized image, window, scale, padding, crop | Resizes an image keeping the aspect ratio unchanged
resize_mask(mask, scale, padding, crop=None) | | Resizes a mask using the given scale and padding. Typically, you get the scale and padding from resize_image() to ensure both, the image and the mask, are resized consistently
minimize_mask(bbox, mask, mini_shape) | | Resize masks to a smaller version to reduce memory load. Mini-masks can be resized back to image scale using expand_masks()
expand_mask(bbox, mini_mask, image_shape) | ndarray mask | Resizes mini masks back to image size. Reverses the change of minimize_mask()
unmold_mask(mask, bbox, image_shape) | | Converts a mask generated by the neural network (floating-point value in the range [0, 1]) to a format similar to its original shape. mask: [height, width] of type float. A small, typically 28x28 mask. bbox: [y1, x1, y2, x2]. The box to fit the mask in
generate_anchors(scales, ratios, shape, feature_stride, anchor_stride) | | scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]. ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]. shape: [height, width] spatial shape of the feature map over which to generate anchors. feature_stride: Stride of the feature map relative to the image in pixels. anchor_stride: Stride of anchors on the feature map. For example, if the value is 2 then generate anchors for every other feature map pixel.
generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride) | anchors: [N, (y1, x1, y2, x2)] | Generate anchors at different levels of a feature pyramid. Each scale is associated with a level of the pyramid, but each ratio is used in all levels of the pyramid. All generated anchors in one array. Sorted with the same order of the given scales. So, anchors of scale[0] come first, then anchors of scale[1], and so on.
trim_zeros(x) | x or reduced x | It's common to have tensors larger than the available data and pad with zeros. This function removes rows that are all zeros. [rows, columns]
compute_matches(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, iou_threshold=0.5, score_threshold=0.0) | tuple (gt_match, pred_match, overlaps) | Finds matches between prediction and ground truth instances
compute_ap(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, iou_threshold=0.5) | (mAP, precisions, recalls, overlaps) | Compute Average Precision at a set IoU threshold (default 0.5). Returns: mAP: Mean Average Precision. precisions: List of precisions at different class score thresholds. recalls: List of recall values at different class score thresholds. overlaps: [pred_boxes, gt_boxes] IoU overlaps
compute_ap_range(gt_box, gt_class_id, gt_mask, pred_box, pred_class_id, pred_score, pred_mask, iou_thresholds=None, verbose=1) | AP | Compute AP over a range or IoU thresholds. Default range is 0.5-0.95.
compute_recall(pred_boxes, gt_boxes, iou) | (recall, positive_ids) | Compute the recall at the given IoU threshold. It's an indication of how many GT boxes were found by the given prediction boxes. pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates. 
batch_slice(inputs, graph_fn, batch_size, names=None) | | Splits inputs into slices and feeds each slice to a copy of the given computation graph and then combines the results. It allows you to run a graph on a batch of inputs even if the graph is written to support one instance only. inputs: list of tensors. All must have the same first dimension length. graph_fn: A function that returns a TF tensor that's part of a graph. batch_size: number of slices to divide the data into. names: If provided, assigns names to the resulting tensors. This a temporary solution
download_trained_weights(coco_model_path, verbose=1) | | Download COCO trained weights from Releases. 
norm_boxes(boxes, shape) | [N, (y1, x1, y2, x2)] in normalized coordinates | Converts boxes from pixel coordinates to normalized coordinates. boxes: [N, (y1, x1, y2, x2)] in pixel coordinates. shape: [..., (height, width)] in pixels. In pixel coordinates (y2, x2) is outside the box. But in normalized coordinates it's inside the box
denorm_boxes(boxes, shape) |[N, (y1, x1, y2, x2)] in pixel coordinates | Converts boxes from normalized coordinates to pixel coordinates. boxes: [N, (y1, x1, y2, x2)] in normalized coordinates. shape: [..., (height, width)] in pixels. Note: In pixel coordinates (y2, x2) is outside the box. But in normalized coordinates it's inside the box.
resize(image, output_shape, order=1, mode='constant', cval=0, clip=True, preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None) | same as from skimage.transform.resize() | A wrapper for Scikit-Image resize(). Scikit-Image generates warnings on every call to resize() if it doesn't receive the right parameters. The right parameters depend on the version of skimage. This solves the problem by using different parameters per version. And it provides a central place to control resizing defaults.
