import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import os

## Load model ----

from zooprocess_multiple_separator import api
reload(api)
api.warm()
# the variables are loaded inside the api modules. get some equivalent here.
model=api.model
processor=api.processor
device=api.device


## Test underlying functions ----
from zooprocess_multiple_separator import utils
reload(utils)

image_paths = ["../test_images/m_1245.jpg", "../test_images/s_0201.jpg"]
image_names = [os.path.basename(img) for img in image_paths]

outputs = utils.predict_panoptic_masks(image_paths, image_names, model, processor, device, bottom_crop)
outputs

from zooprocess_multiple_separator import utils
reload(utils)
masks, score, image, binary_image = \
    utils.predict_panoptic_masks(image_path=test_img,
    model=model, processor=processor, device=device, min_mask_score=0.9, bottom_crop=0)
print(score)
plt.clf(); plt.imshow(image, cmap='Greys_r', interpolation='none'); plt.show()
plt.clf(); plt.imshow(binary_image, cmap='Greys_r', interpolation='none'); plt.show()
plt.clf(); plt.imshow(masks); plt.show()
sep_lines = utils.separate_with_watershed(masks, binary_image)
plt.clf(); plt.imshow(sep_lines); plt.show()

# encode sep lines
sep_coords = np.where(sep_lines==1)
sep_coords = tuple([sep.tolist() for sep in sep_coords])
shape = sep_lines.shape

# reconstruct the separation image
X = np.zeros(shape)
X[sep_coords] = 1
plt.clf(); plt.imshow(X); plt.show()


## Test API ----

# mimick the input object
class Input():
  def __init__(self, filename=None):
    self.filename=None
    self.original_filename=None
    self.content_type=None
input_data = Input()
input_data.filename = "../test_images/Archive.zip"
input_data.content_type = 'application/zip'

res = api.predict(images=input_data, min_mask_score=0.9, bottom_crop=0)
# X = np.zeros(res['image_shape'])
# X[tuple(res['separation_coordinates'])] = 1
# plt.clf(); plt.imshow(X); plt.show()
# res['score']
