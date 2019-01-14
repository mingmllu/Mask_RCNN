## Notes about the file model.py

### Utility Functions

Functions               | Details
----------------------- | ----------------------------------------------------------------------------------------------
log(text, array=None) | Prints a text message. And, optionally, if a Numpy array is provided it prints it's shape, min, and max values
compute_backbone_shapes(config, image_shape) | Check config.BACKBONE. If it is callable, return config.COMPUTE_BACKBONE_SHAPE(image_shape). Otherwise, it must be "resnet50", or "resnet101". Computes the width and height of each stage of the backbone network. Returns: [N, (height, width)]. Where N is the number of stages.

### Resnet Graph

Functions               | Details
----------------------- | ----------------------------------------------------------------------------------------------
identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True) | The identity_block is the block that has no conv layer at shortcut. input_tensor: input tensor. kernel_size: default 3, the kernel size of middle conv layer at main path. filters: list of integers, the nb_filters [the number of generated feature maps] of 3 conv layer at main path. stage: integer, current stage label, used for generating layer names. block: 'a','b'..., current block label, used for generating layer names. use_bias: Boolean. To use or not use a bias in conv layers. train_bn: Boolean. Train or freeze Batch Norm layers
conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True) | conv_block is the block that has a conv layer at shortcut. input_tensor: input tensor. kernel_size: default 3, the kernel size of middle conv layer at main path. filters: list of integers, the nb_filters of 3 conv layer at main path. stage: integer, current stage label, used for generating layer names. block: 'a','b'..., current block label, used for generating layer names. use_bias: Boolean. To use or not use a bias in conv layers. train_bn: Boolean. Train or freeze Batch Norm layers. Note that from stage 3, the first conv layer at main path is with subsample=(2,2) and the shortcut should have subsample=(2,2) as well
resnet_graph(input_image, architecture, stage5=False, train_bn=True) | Build a ResNet graph. architecture: Can be resnet50 or resnet101. stage5: Boolean. If False, stage5 of the network is not created. train_bn: Boolean. Train or freeze Batch Norm layers. Return: [C1, C2, C3, C4, C5] with each element corresponding to the output of convolutional stage


### Proposal Layer

Functions               | Details
----------------------- | ----------------------------------------------------------------------------------------------
apply_box_deltas_graph(boxes, deltas) | Applies the given deltas to the given boxes. boxes: [N, (y1, x1, y2, x2)] boxes to update. deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply.
clip_boxes_graph(boxes, window) | boxes: [N, (y1, x1, y2, x2)] window: [4] in the form y1, x1, y2, x2



### Classes

#### class BatchNorm(keras.layers.BatchNormalization)

Extends the Keras BatchNormalization class to allow a central place to make changes if needed. Batch normalization has a negative effect on training if batches are small so this layer is often frozen (via setting in Config class) and functions as linear layer

Nothing is done currently in the class.

#### class ProposalLayer(keras.engine.Layer)

Receives anchor scores and selects a subset to pass as proposals to the second stage. Filtering is done based on anchor scores and non-max suppression to remove overlaps. It also applies bounding box refinement deltas to anchors.
   
Inputs:
* rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
* rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
* anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

Returns:
* Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]

Methods                 | Details
----------------------- | ----------------------------------------------------------------------------------------------
__init__(self, proposal_count, nms_threshold, config=None, **kwargs) | Initialization
call(self, inputs) | This method is invoked via operator (). The inputs include box scores (inputs[0][:, :, 1]), box deltas (inputs[1]) and anchors (inputs[2]). Within this method, the following TF methods are used: tf.minimum(), tf.nn.top_k(), tf.gather(), tf.image.non_max_suppression(), tf.maximum(), tf.pad(). 
compute_output_shape(self, input_shape) | (None, self.proposal_count, 4)

### ROIAlign Layer

#### class PyramidROIAlign(keras.engine.Layer)

Implements ROI Pooling on multiple levels of the feature pyramid.

Methods                 | Details
----------------------- | ----------------------------------------------------------------------------------------------
__init__(self, pool_shape, **kwargs) | pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
call(self, inputs) | boxes (inputs[0]): [batch, num_boxes, (y1, x1, y2, x2)] in normalized coordinates. Possibly padded with zeros if not enough boxes to fill the array. image_meta (inputs[1]): [batch, (meta data)] Image details. See compose_image_meta(). feature_maps (inputs[2:]): List of feature maps from different levels of the pyramid. Each is [batch, height, width, channels]. tf.cast(), tf.minimum(), tf.round(), tf.squeeze(), tf.where(), tf.equal(), tf.gather_nd(),tf.stop_gradient(), tf.concat(), tf.expand_dims(), tf.shape(), tf.range(), tf.nn.top_k(), tf.gather(), tf.reshape()
compute_output_shape(self, input_shape) | Returns input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )

### Detection Target Layer

Functions               | Details
----------------------- | ----------------------------------------------------------------------------------------------
overlaps_graph(boxes1, boxes2) | Computes IoU overlaps between two sets of boxes. boxes1, boxes2: [N, (y1, x1, y2, x2)]
detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config) | Generates detection targets for one image. Subsamples proposals and generates target class IDs, bounding box deltas, and masks for each. tf.Assert(), tf.control_dependencies(), tf.identity(), tf.boolean_mask(), tf.random_shuffle(), tf.argmax(), tf.image.crop_and_resize(), tf.range(), tf.pad(), 

#### class DetectionTargetLayer(keras.engine.Layer)

Subsamples proposals and generates target box refinement, class_ids, and masks for each

Methods                 | Details
----------------------- | ----------------------------------------------------------------------------------------------
__init__(self, config, **kwargs) | Stored the config table
call(self, inputs) | proposals = inputs[0], gt_class_ids = inputs[1], gt_boxes = inputs[2], gt_masks = inputs[3]
compute_output_shape(self, input_shape) | 
compute_mask(self, inputs, mask=None) | return [None, None, None, None]

### Detection Layer

Functions               | Details
----------------------- | ----------------------------------------------------------------------------------------------
refine_detections_graph(rois, probs, deltas, window, config) | Refine classified proposals and filter overlaps and return final detections. tf.argmax(), tf.stack(), tf.gather_nd(), tf.where(), tf.sets.set_intersection(), tf.expand_dims(), tf.sparse_tensor_to_dense(),  tf.unique(),
tf.image.non_max_suppression(), tf.map_fn(), 

#### class DetectionLayer(keras.engine.Layer)

Takes classified proposal boxes and their bounding box deltas and returns the final detection boxes.

Returns: [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where coordinates are normalized

Methods                 | Details
----------------------- | ----------------------------------------------------------------------------------------------
__init__(self, config=None, **kwargs) |
call(self, inputs) |
compute_output_shape(self, input_shape) | Return (None, self.config.DETECTION_MAX_INSTANCES, 6)

### Region Proposal Network (RPN)

Functions               | Details
----------------------- | ----------------------------------------------------------------------------------------------
rpn_graph(feature_map, anchors_per_location, anchor_stride) | Builds the computation graph of Region Proposal Network. feature_map: backbone features [batch, height, width, depth]. anchors_per_location: number of anchors per pixel in the feature map. anchor_stride: Controls the density of anchors. Typically 1 (anchors for every pixel in the feature map), or 2 (every other pixel). Returns: rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax). rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities. rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors
build_rpn_model(anchor_stride, anchors_per_location, depth) | Builds a Keras model of the Region Proposal Network. It wraps the RPN graph so it can be used multiple times with shared weights. KL.Input(), KL.Model()

### Feature Pyramid Network Heads

Functions               | Details
----------------------- | ----------------------------------------------------------------------------------------------
fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size=1024) | Builds the computation graph of the feature pyramid network classifier and regressor heads. KL.TimeDistributed()
build_fpn_mask_graph(rois, feature_maps, image_meta, pool_size, num_classes, train_bn=True) | Builds the computation graph of the mask head of Feature Pyramid Network

### Loss Functions

Functions               | Details
----------------------- | ----------------------------------------------------------------------------------------------
smooth_l1_loss(y_true, y_pred) | Implements Smooth-L1 loss. y_true and y_pred are typically: [N, 4], but could be any shape
rpn_class_loss_graph(rpn_match, rpn_class_logits) | RPN anchor classifier loss
rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox) | Return the RPN bounding box loss graph. keras.backend.switch()
mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids) | Loss for the classifier head of Mask RCNN. tf.reduce_sum()
mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox) | Loss for Mask R-CNN bounding box refinement
mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks) | Mask binary cross-entropy loss for the masks head


### Data Generator

Functions               | Details
----------------------- | ----------------------------------------------------------------------------------------------
load_image_gt(dataset, config, image_id, augment=False, augmentation=None, use_mini_mask=False) | Load and return ground truth data for an image (image, mask, bounding boxes)
build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config) | Generate targets for training Stage 2 classifier and mask heads. This is not used in normal training. It's useful for debugging or to train the Mask RCNN heads without using the RPN head.
build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config) | Given the anchors and GT boxes, compute overlaps and identify positive anchors and deltas to refine them to match their corresponding GT boxes.
generate_random_rois(image_shape, count, gt_class_ids, gt_boxes) | Generates ROI proposals similar to what a region proposal network would generate.
data_generator(dataset, config, shuffle=True, augment=False, augmentation=None, random_rois=0, batch_size=1, detection_targets=False,  no_augmentation_sources=None) | A Python generator that returns images and corresponding target class ids, bounding box deltas, and masks. Upon calling next() on it, the generator returns two lists, inputs and outputs. The contents of the lists differs depending on the received arguments


### MaskRCNN Class

Encapsulates the Mask RCNN model functionality. The actual Keras model is in the keras_model property.

Methods                 | Details
----------------------- | ----------------------------------------------------------------------------------------------
__init__(self, mode, config, model_dir) | 
build(self, mode, config) | Build Mask R-CNN architecture. input_shape: The shape of the input image. mode: Either "training" or "inference". The inputs and outputs of the model differ accordingly
find_last(self) | Finds the last checkpoint file of the last trained model in the model directory. Returns: The path of the last checkpoint file
load_weights(self, filepath, by_name=False, exclude=None) | Modified version of the corresponding Keras function with the addition of multi-GPU support and the ability to exclude some layers from loading. exclude: list of layer names to exclude
get_imagenet_weights(self) | Downloads ImageNet trained weights from Keras. Returns path to weights file.
compile(self, learning_rate, momentum) | Gets the model ready for training. Adds losses, regularization, and metrics. Then calls the Keras compile() function. keras_model.compile()
set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1) | Sets model layers as trainable if their names match the given regular expression
set_log_dir(self, model_path=None) | Sets the model log directory and epoch counter
train(self, train_dataset, val_dataset, learning_rate, epochs, layers, augmentation=None, custom_callbacks=None, no_augmentation_sources=None) | Train the model. keras.callbacks.TensorBoard(), keras.callbacks.ModelCheckpoint(), self.keras_model.fit_generator()
mold_inputs(self, images) | Takes a list of images and modifies them to the format expected as an input to the neural network. images: List of image matrices [height,width,depth]. Images can have different sizes.
unmold_detections(self, detections, mrcnn_mask, original_image_shape, image_shape, window) | Reformats the detections of one image from the format of the neural network output to a format suitable for use in the rest of the application.
detect(self, images, verbose=0) | Runs the detection pipeline.
detect_molded(self, molded_images, image_metas, verbose=0) | Runs the detection pipeline, but expect inputs that are molded already. Used mostly for debugging and inspecting the model.
get_anchors(self, image_shape) | Returns anchor pyramid for the given image size 
ancestor(self, tensor, name, checked=None) | Finds the ancestor of a TF tensor in the computation graph
find_trainable_layer(self, layer) | If a layer is encapsulated by another layer, this function digs through the encapsulation and returns the layer that holds the weights.
get_trainable_layers(self) | Returns a list of layers that have weights.
run_graph(self, images, outputs, image_metas=None) | Runs a sub-set of the computation graph that computes the given outputs.

### Data Formatting

Functions               | Details
----------------------- | ----------------------------------------------------------------------------------------------
compose_image_meta(image_id, original_image_shape, image_shape, window, scale, active_class_ids) | Takes attributes of an image and puts them in one 1D array.
parse_image_meta(meta) | Parses an array that contains image attributes to its components. See compose_image_meta() for more details.
parse_image_meta_graph(meta) | Parses a tensor that contains image attributes to its components. See compose_image_meta() for more details.
mold_image(images, config) | Expects an RGB image (or array of images) and subtracts the mean pixel and converts it to float. Expects image colors in RGB order.
unmold_image(normalized_images, config) | Takes a image normalized with mold() and returns the original


### Miscellenous Graph Functions

Functions               | Details
----------------------- | ----------------------------------------------------------------------------------------------
trim_zeros_graph(boxes, name='trim_zeros') | Often boxes are represented with matrices of shape [N, 4] and are padded with zeros. This removes zero boxes.
batch_pack_graph(x, counts, num_rows) | Picks different number of values from each row in x depending on the values in counts.
norm_boxes_graph(boxes, shape) | Converts boxes from pixel coordinates to normalized coordinates
denorm_boxes_graph(boxes, shape) | Converts boxes from normalized coordinates to pixel coordinates







