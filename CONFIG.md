## Notes

The configuration parameters are grouped into a class "Config" that is derived from Python "object".

### Configurable Parameters

Parameters (Properties)            | Default Value | Connments
---------------------------------- | ------------- | -------------------------------------------------------------
NAME | None | Override in sub-classes
GPU_COUNT | 1 | NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
IMAGES_PER_GPU | 2 | Number of images to train with on each GPU. A 12GB GPU can typically handle 2 images of 1024x1024px. Adjust based on your GPU memory and image sizes. Use the highest number that your GPU can handle for best performance.
STEPS_PER_EPOCH | 1000 | Number of training steps per epoch. This doesn't need to match the size of the training set. Tensorboard updates are saved at the end of each epoch, so setting this to a smaller number means getting more frequent TensorBoard updates. Validation stats are also calculated at each epoch end and they might take a while, so don't set this too small to avoid spending a lot of time on validation stats.
VALIDATION_STEPS | 50 | Number of validation steps to run at the end of every training epoch. | A bigger number improves accuracy of validation stats, but slows down the training.
BACKBONE | "resnet101" | Backbone network architecture. Supported values are: resnet50, resnet101. You can also provide a callable that should have the signature of model.resnet_graph. If you do so, you need to supply a callable to COMPUTE_BACKBONE_SHAPE as well
COMPUTE_BACKBONE_SHAPE | None | Only useful if you supply a callable to BACKBONE. Should compute the shape of each layer of the FPN Pyramid. See model.compute_backbone_shapes
BACKBONE_STRIDES | [4, 8, 16, 32, 64] | The strides of each layer of the FPN (Feature Pyramid Network). These values are based on a Resnet101 backbone.
FPN_CLASSIF_FC_LAYERS_SIZE | 1024 | Size of the fully-connected layers in the classification graph
TOP_DOWN_PYRAMID_SIZE | 256 | Size of the top-down layers used to build the feature pyramid
NUM_CLASSES | 1  | Number of classification classes (including background). Override in sub-classes
RPN_ANCHOR_SCALES | (32, 64, 128, 256, 512) | Length of square anchor side in pixels. For Region proposal network
RPN_ANCHOR_RATIOS | [0.5, 1, 2] | Ratios of anchors at each cell (width/height). A value of 1 represents a square anchor, and 0.5 is a wide anchor
RPN_ANCHOR_STRIDE | 1 | Anchor stride. If 1 then anchors are created for each cell in the backbone feature map. If 2, then anchors are created for every other cell, and so on.
RPN_NMS_THRESHOLD | 0.7 | Non-max suppression threshold to filter RPN proposals. You can increase this during training to generate more propsals.
RPN_TRAIN_ANCHORS_PER_IMAGE | 256 | How many anchors per image to use for RPN training
PRE_NMS_LIMIT | 6000 | ROIs kept after tf.nn.top_k and before non-maximum suppression
POST_NMS_ROIS_TRAINING | 2000 | ROIs kept after non-maximum suppression (training)
POST_NMS_ROIS_INFERENCE | 1000 | ROIs kept after non-maximum suppression (inference)
USE_MINI_MASK | True | If enabled, resizes instance masks to a smaller size to reduce memory load. Recommended when using high-resolution images.
MINI_MASK_SHAPE | (56, 56)  | (height, width) of the mini-mask. A mini-mask is used to store the original mask to save memory space. This will cause a certain accuracy loss.
IMAGE_RESIZE_MODE | "square" | Control input image resizing.  Generally, use the "square" resizing mode for training and predicting and it should work well in most cases. In this mode, images are scaled up such that the small side is = IMAGE_MIN_DIM, but ensuring that the scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is padded with zeros to make it a square so multiple images can be put in one batch. Available resizing modes: "none":   No resizing or padding. Return the image unchanged. "square": Resize and pad with zeros to get a square image of size [max_dim, max_dim]. "pad64":  Pads width and height with zeros to make them multiples of 64. If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales up before padding. IMAGE_MAX_DIM is ignored in this mode. The multiple of 64 is needed to ensure smooth scaling of feature maps up and down the 6 levels of the FPN pyramid (2**6=64). "crop":   Picks random crops from the image. First, scales the image based on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only. IMAGE_MAX_DIM is not used in this mode.
IMAGE_MIN_DIM ??? | 800 | If images are resized (i.e., IMAGE_RESIZE_MODE is not None), the short side will be IMAGE_MIN_DIM as long as the long side is not exceeding IMAGE_MAX_DIM
IMAGE_MAX_DIM ??? | 1024 |
IMAGE_MIN_SCALE ??? | 0 | Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further up scaling. For example, if set to 2 then images are scaled up to double the width and height, or more, even if MIN_IMAGE_DIM doesn't require it. Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
IMAGE_CHANNEL_COUNT | 3 | Number of color channels per image. RGB = 3, grayscale = 1, RGB-D (3 color + 1 depth channels) = 4. Changing this requires other changes in the code. See the WIKI for more details: https://github.com/matterport/Mask_RCNN/wiki
MEAN_PIXEL | np.array([123.7, 116.8, 103.9]) | Image mean (RGB)
TRAIN_ROIS_PER_IMAGE | 200 | Number of ROIs per image to feed to classifier/mask heads. The Mask RCNN paper uses 512 but often the RPN doesn't generate enough positive proposals to fill this and keep a positive:negative ratio of 1:3. You can increase the number of proposals by adjusting  the RPN NMS threshold.
ROI_POSITIVE_RATIO | 0.33 | Percent of positive ROIs used to train classifier/mask heads
POOL_SIZE | 7 | Pooled ROIs
MASK_POOL_SIZE | 14 |
MASK_SHAPE ??? | [28, 28] | Shape of output mask. To change this you also need to change the neural network mask branch
MAX_GT_INSTANCES | 100 | Maximum number of ground truth instances to use in one image
RPN_BBOX_STD_DEV | np.array([0.1, 0.1, 0.2, 0.2]) | Bounding box refinement standard deviation for RPN and final detections.
BBOX_STD_DEV | np.array([0.1, 0.1, 0.2, 0.2]) |
DETECTION_MAX_INSTANCES | 100 | Max number of final detections
DETECTION_MIN_CONFIDENCE | 0.7 | Minimum probability value to accept a detected instance. ROIs below this threshold are skipped
DETECTION_NMS_THRESHOLD | 0.3 | Non-maximum suppression threshold for detection
LEARNING_RATE | 0.001 | Learning rate. The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes weights to explode. Likely due to differences in optimizer implementation.
LEARNING_MOMENTUM | 0.9 | Learning momentum
WEIGHT_DECAY | 0.0001 | Weight decay regularization
LOSS_WEIGHTS | {"rpn_class_loss": 1.,"rpn_bbox_loss": 1., "mrcnn_class_loss": 1.,"mrcnn_bbox_loss": 1., "mrcnn_mask_loss": 1. } | Loss weights for more precise optimization. Can be used for R-CNN training setup.
USE_RPN_ROIS | True | Use RPN ROIs or externally generated ROIs for training. Keep this True for most situations. Set to False if you want to train the head branches on ROI generated by code rather than the ROIs from the RPN. For example, to debug the classifier head without having to train the RPN.
TRAIN_BN | False  | Train or freeze batch normalization layers. None: Train BN layers. This is the normal mode. False: Freeze BN layers, Good when using a small batch size. True: (don't use). Set layer in training mode even when predicting. Defaulting to False since batch size is often small.
GRADIENT_CLIP_NORM | 5.0 | Gradient norm clipping
    
### Computed Properties

Computed Properties                | Definition    | Connments
---------------------------------- | ------------- | -------------------------------------------------------------
BATCH_SIZE | IMAGES_PER_GPU * GPU_COUNT | Effective batch size
IMAGE_SHAPE | depending on IMAGE_RESIZE_MODE | Input image size. If IMAGE_RESIZE_MODE == "crop", then IMAGE_SHAPE = np.array([IMAGE_MIN_DIM,IMAGE_MIN_DIM, IMAGE_CHANNEL_COUNT]), else IMAGE_SHAPE = np.array([IMAGE_MAX_DIM, IMAGE_MAX_DIM, IMAGE_CHANNEL_COUNT])
IMAGE_META_SIZE | 1 + 3 + 3 + 4 + 1 + NUM_CLASSES | Image meta data length. See compose_image_meta() for details.
