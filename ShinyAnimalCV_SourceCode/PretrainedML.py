#import dependencies 
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import pandas as pd

# turn off tf log 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# set up hyper-parameters
class ShapesConfig(Config):
    """Configuration for training on the pigs dataset.
    Derives from the base Config class and overrides values specific
    to the pigs dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ROI" 
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE = "resnet50"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + stem ##

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGE_MIN_DIM = 400
    
config = ShapesConfig()
config.display()

# define get_ax for plot visualization
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# set configurations 
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9
    USE_MINI_MASK = False

inference_config = InferenceConfig()

