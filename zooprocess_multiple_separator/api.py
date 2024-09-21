# -*- coding: utf-8 -*-
"""
Functions to integrate the model with the DEEPaaS API.
To keep this file minimal, functions can be written in a separate file
and only called from here.

To start populating this file, take a look at the docs [1] and at
an exemplar module [2].
[1]: https://docs.ai4os.eu/
[2]: https://github.com/ai4os-hub/ai4os-demo-app
"""

from pathlib import Path
import logging
from webargs import fields

import os
import torch
import torchvision
import numpy as np
from PIL import Image

from zooprocess_multiple_separator import config
from zooprocess_multiple_separator.misc import _catch_error
from zooprocess_multiple_separator import utils

import zipfile
from transformers import Mask2FormerForUniversalSegmentation, MaskFormerImageProcessor

# for development
# import matplotlib.pyplot as plt
# from importlib import reload

# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)

BASE_DIR = Path(__file__).resolve().parents[1]


@_catch_error
def get_metadata():
    """Returns a dictionary containing metadata information about the module.
       DO NOT REMOVE - All modules should have a get_metadata() function

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.API_NAME)
        metadata = {
            "name": config.API_METADATA.get("name"),
            "author": config.API_METADATA.get("authors"),
            "author-email": config.API_METADATA.get("author-emails"),
            "description": config.API_METADATA.get("summary"),
            "license": config.API_METADATA.get("license"),
            "version": config.API_METADATA.get("version"),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        logger.error("Error collecting metadata: %s", err, exc_info=True)
        raise  # Reraise the exception after log


def warm():
    """
    Load model upon application startup
    """
    # make sure the objects are available everywhere after the application starts
    global model, processor, device

    # NB: get the model file from a github release
    model_zip_path = os.path.join(BASE_DIR,
                                 'models',
                                 'learn_plankton_pano_plus5000_8epoch.zip')

    # check that the model file is there, in zipped form
    if not os.path.exists(model_zip_path):
        logger.error("Zipfile of model not found.")
        return None
    
    # if the directory containing the model does not exist,
    # but the zip file does, unzip it
    model_path = model_zip_path[:-4]
    if not os.path.exists(model_path):
        logger.info("Unzipping model files")
        with zipfile.ZipFile(model_zip_path, 'r') as model_zip:
            model_zip.extractall(model_zip_path.strip(model_zip_path.split("/")[-1]))    
    
    # check if a GPU is available, fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model (possibly on GPU)
    logger.info("Loading model")
    processor = MaskFormerImageProcessor.from_pretrained(model_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
    model = model.to(device)


def get_predict_args():
    """
    Get the list of arguments for the predict function
    """
    arg_dict = {
        "image": fields.Field(
            metadata={
                'required': True,
                'type': "file",
                'location': "form",
                'description': "An image containing object(s) to separate."
            }
        ),
        "min_mask_score": fields.Float(
            metadata={
                'required':False,
                'missing': 0.9,
                'description': "The minimum confidence score for a mask to be selected. [Default: 0.9]"
            }
        ),
        "bottom_crop": fields.Int(
            metadata={
                'required':False,
                'missing': 31,
                'description': "Number of pixels to crop from the bottom of the\
                image (e.g. to remove the scale bar). [Default: 31px]"
            }
        )
    }

    return arg_dict


# DEVELOP
# class Img():
#   def __init__(self, filename=None):
#     self.filename=None
# image=Img()
# image.filename="m_0595.jpg"
# 
# api.warm()
# api.predict(image=image, min_mask_score=0.9, bottom_crop=31)

@_catch_error
def predict(**kwargs):
    """
    Compute the white lines to separate objects
    """
    
    # get predicted masks
    mask_sum, mask_centers, img, score = utils.predict_mask_panoptic(
        kwargs['image'].filename,
        model, processor, device,
        kwargs['min_mask_score'], kwargs['bottom_crop']
        )

    # apply watershed algorithm
    # = from each center find connected regions and their separation
    watershed_labels = utils.get_watershed_result(mask_sum, mask_centers, mask=binary_img)
    # io.imshow(watershed_labels); plt.show()

    # compute output separations
    separation_mask = np.zeros_like(watershed_labels)
    separation_mask[watershed_labels == 0] = 1
    # io.imshow(separation_mask); plt.show()

    # # produce a diagnostic plot
    # fig, axes = plt.subplots(nrows=2, ncols=2,
    #                          subplot_kw={'xticks': [], 'yticks': []})
    # 
    # axes[1,0].imshow(img, interpolation='none')
    # axes[1,0].set_title("Original image")
    # 
    # axes[0,0].imshow(mask_sum, cmap="viridis")
    # axes[0,0].set_title("Sum of predicted masks")
    # 
    # axes[0,1].imshow(watershed_labels, interpolation='none')
    # axes[0,1].set_title("Watershed result")
    # 
    # axes[1,1].imshow(separation_mask, cmap='Greys_r', interpolation='none')
    # axes[1,1].set_title("Extracted line(s)")
    # 
    # plt.show()

    return separation_mask, str(score)


# def get_train_args():
#     return {}
#
#
# def train(**kwargs):
#     return None
