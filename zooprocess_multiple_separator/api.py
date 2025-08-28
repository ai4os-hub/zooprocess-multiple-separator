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
# import torchvision
import numpy as np

from zooprocess_multiple_separator import config
from zooprocess_multiple_separator.misc import _catch_error
from zooprocess_multiple_separator import utils
from zooprocess_multiple_separator import train as train_file

from glob import glob

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

    # list all models and pick the latest
    # NB: we only warm-load this one for efficiency purposes
    model_root = os.path.join(BASE_DIR, 'models', '*')
    model_paths = [p for p in glob(model_root) if p.endswith('.zip')]
    model_paths.sort()
    model_zip_path = model_paths[-1]
    # NB: get at least one model file from a github release, as per the README
    
    # if not os.path.exists(model_path):
    #     print("Model not found.")
    # print("Load model ", model_path)

    # NB: get the model file from a github release
    # model_zip_path = os.path.join(BASE_DIR,
    #                              'models',
    #                              'learn_plankton_pano_plus5000_8epoch.zip')

    # check that the model file is there, in zipped form
    if not os.path.exists(model_zip_path):
        logger.error("Zip file of model not found.")
        return None
    
    # if the directory containing the model does not exist,
    # but the zip file does, unzip it
    model_path = model_zip_path[:-4]
    print("Model path", model_path)
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
        "images": fields.Raw(
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
       ),
      "max_prop_missing": fields.Float(
            metadata={
                'description': "A proportion of the area original object. Any\
                region of the original object(s) that is missed by the panoptic\
                segmenter and is larger than max_prop_missing * original_area\
                will be considered as a potential new object. [Default: 0.2]"
            },
            required=False,
            load_default=0.2
       )
    }

    return arg_dict


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
        print("data zip filename", data.filename)
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
            kwargs['min_mask_score'], kwargs['bottom_crop'], kwargs['max_prop_missing']
            )
    
        # apply watershed algorithm
        # = from each center find connected regions and their separation
        sep_lines = utils.separate_with_watershed(masks, binary_image)
        # NB: this has 0 as the background and 1 where the separation lines should be drawn

    
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

    arg_dict = {
        "n_epochs": fields.Int(
            metadata={
                'description': "Number of training epochs. [Default: 10]"
            },
            required=False,
            missing=10
        ),
        "batch_size": fields.Int(
            metadata={
                'description': "Number of images in each training batch. Should\
                probably be as large as the GPU allows. [Default: 128]"
            },
            required=False,
            load_default=16,
        ),
        "list_hashtags": fields.List(
            fields.Str(),
            metadata={
                'description': "List of hashtags to train on. [Default: ['plancton']]"
            },
            required=False,
            load_default=["plancton"]
        ),

    }
    
    return arg_dict


#
# def train(**kwargs):
#     return None

def train(**kwargs):

    results=[]
    try:

        list_loss=train_file.run_train(
      data_dir=os.path.join(BASE_DIR, 'data'),
      out_dir=os.path.join(BASE_DIR, 'models'),
      device=device,
      n_epochs=kwargs['n_epochs'],
        list_hashtags=kwargs['list_hashtags'],
      batch_size=kwargs['batch_size'],
    )
        print("api return")
        print(list_loss)
        results.append({"loss": list_loss})
    except Exception as e:
        print(str(e))
        logger.error("Error during training: %s", e, exc_info=True)
    print("en dehors du try except")

    return None
