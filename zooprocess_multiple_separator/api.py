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
from marshmallow import Schema

import os
import torch
import torchvision
import numpy as np

from zooprocess_multiple_separator import config
from zooprocess_multiple_separator.misc import _catch_error
from zooprocess_multiple_separator import utils

import zipfile
from transformers import Mask2FormerForUniversalSegmentation, MaskFormerImageProcessor

# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)

BASE_DIR = Path(__file__).resolve().parents[1]

# initialise global variables
model = None
processor = None
device = None

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

    # check if a GPU is available, fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NB: get the model file from a github release
    model_zip_path = os.path.join(BASE_DIR,
                                 'models',
                                 'learn_plankton_pano_plus5000_8epoch.zip')

    # check that the model file is there, in zipped form
    if not os.path.exists(model_zip_path):
        logger.error("Zip file of model not found.")
        return None
    
    # if the directory containing the model does not exist,
    # but the zip file does, unzip it
    model_path = model_zip_path[:-4]
    if not os.path.exists(model_path):
        logger.info("Unzipping model files")
        with zipfile.ZipFile(model_zip_path, 'r') as model_zip:
            model_zip.extractall(model_zip_path.strip(model_zip_path.split("/")[-1]))    

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
        "images": fields.Field(
            metadata={
                'type': "file",
                'location': "form",
                'description': "A zip file containing the images to classify (all\
                images should be at the root of the zip file) or a single image file."
            },
            required=True
        ),
       "min_mask_score": fields.Float(
            metadata={
                'description': "The minimum confidence score for a mask to be\
                selected. [Default: 0.9]"
            },
            required=False,
            load_default=0.9
       ),
       "bottom_crop": fields.Int(
            metadata={
                'description': "Number of pixels to crop from the bottom of the\
                image (e.g. to remove the scale bar). [Default: 31px]"
            },
            required=False,
            load_default=31,
       )
    }

    return arg_dict


# class OutputSchema(Schema):
#     name = fields.Str()
#     separation_coordinates = fields.Tuple(
#         (fields.List(fields.Int()),fields.List(fields.Int())),
#         required=True,
#         metadata={'description': """a list containing two other lists of ints:\
#         the x and y coordinates of pixels that draw lines on the original image,\
#         to separate multiple organisms. This list can be used to subset 2D arrays.\
#         For example, to create a black image with white separation lines, one can write:
#       import numpy as np
#       X = np.zeros(image_shape)
#       X[separation_coordinates] = 1
#       """}
#     )
#     image_shape = fields.Tuple(
#         (fields.Int(),fields.Int()),
#         required=True,
#         metadata={'description': "the height and width of the original image."}
#     )
#     score = fields.Float(
#         required=True,
#         metadata={'description': "an estimate of the confidence of the network\
#         for the quality of separation; this is very appropximate (and in [0,1])."}
#     )
# 
# schema = fields.List(fields.Nested(OutputSchema))

@_catch_error
def predict(**kwargs):
    """
    Compute the white lines to separate objects
    
    Args:
        See get_predict_args() above.
    
    Returns:
        See schema, above.
    """
    
    import tempfile
    data = kwargs['images']

    # get input files
    # either as a zip
    if data.content_type == 'application/zip':
        # extract
        tmp_input = tempfile.mkdtemp()
        with zipfile.ZipFile(data.filename, 'r') as zip_file:
            zip_file.extractall(tmp_input)
        # keep only images
        filenames = sorted(os.listdir(tmp_input))
        filenames = [file for file in filenames if file.endswith(('jpg', 'png', 'jpeg'))]
        filepaths = [os.path.join(tmp_input, name) for name in filenames]
    # or as a single item
    else:
        filepaths = [data.filename]
        filenames = [data.original_filename]

    results = []
    for i in range(len(filepaths)):

        # get predicted masks
        masks, score, image, binary_image = utils.predict_panoptic_masks(
            filepaths[i],
            model, processor, device,
            kwargs['min_mask_score'], kwargs['bottom_crop']
            )
    
        # apply watershed algorithm
        # = from each center find connected regions and their separation
        sep_lines = utils.separate_with_watershed(masks, binary_image)
        # NB: this has 0 as the background and 1 where the separation lines should be drawn
    
        # # produce a diagnostic plot
        # fig, axes = plt.subplots(nrows=2, ncols=2,
        #                          subplot_kw={'xticks': [], 'yticks': []})
        # 
        # axes[0,0].imshow(image, cmap='Greys_r', interpolation='none')
        # axes[0,0].set_title("Original image (cropped)")
        # 
        # axes[0,1].imshow(masks, interpolation='none')
        # axes[0,1].set_title("Predicted masks")
        # 
        # axes[1,1].imshow(sep_lines, interpolation='none')
        # axes[1,1].set_title("Watershed result")
        # 
        # axes[1,0].imshow(image, cmap='Greys_r', interpolation='none')
        # from matplotlib.colors import ListedColormap
        # my_cmap = ListedColormap(colors='red')
        # my_cmap.set_under('k', alpha=0)
        # axes[1,0].imshow(sep_lines, cmap=my_cmap, interpolation='none', clim=[0.1,10])
        # axes[1,0].set_title("Final separation")
        # 
        # plt.show()
    
        # encode the lines to draw as a sparse image
        sep_coords = np.where(sep_lines==1)
        sep_coords = tuple([sep.tolist() for sep in sep_coords])
        shape = sep_lines.shape
        
        results.append({
            "name": filenames[i],
            "separation_coordinates": sep_coords,
            "image_shape": shape,
            "score": score
        })

    return results


# uncomment to make deepaas-cli working
def get_train_args():
    return {}

#
# def train(**kwargs):
#     return None
