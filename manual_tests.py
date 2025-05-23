import matplotlib.pyplot as plt
import numpy as np
from importlib import reload

def imshow(x):
  from matplotlib import pyplot as plt
  plt.imshow(x.permute(1, 2, 0))
  plt.show()


## Load model ----

from zooprocess_multiple_separator import api
reload(api)
api.warm()
model=api.model
processor=api.processor
device=api.device


## Test underlying functions ----
image_path = "../test_images/m_1245.jpg"
image_path = "/home/jiho/datasets/juday/images/612787652.jpg"

from zooprocess_multiple_separator import utils
reload(utils)
masks, score, image, binary_image = \
    utils.predict_panoptic_masks(image_path=image_path,
    model=model, processor=processor, device=device, min_mask_score=0.9, bottom_crop=31)
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
