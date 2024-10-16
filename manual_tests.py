# DEVELOP
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload

from zooprocess_multiple_separator import api
reload(api)
model, processor, device = api.warm()

# Test underlying functions
from zooprocess_multiple_separator import utils
reload(utils)
masks, score, image = \
    utils.predict_panoptic_masks(image_path="0be42a52fc61857c6a83eec1fffe485f_38088162.png",
    model=model, processor=processor, device=device, min_mask_score=0.9, bottom_crop=0)
print(score)
plt.clf(); plt.imshow(masks); plt.show()
plt.clf(); plt.imshow(image); plt.show()
sep_lines = utils.separate_with_watershed(panoptic_masks)
plt.clf(); plt.imshow(sep_lines); plt.show()

# encode sep lines
sep_coords = np.where(sep_lines==1)
sep_coords = [sep.tolist() for sep in sep_coords]
shape = sep_lines.shape

# reconstruct image
X = np.zeros(shape)
X[tuple(sep_coords)] = 1
plt.clf(); plt.imshow(X); plt.show()

# Test API
class Img():
  def __init__(self, filename=None):
    self.filename=None
image=Img()
image.filename="0be42a52fc61857c6a83eec1fffe485f_38088162.png"

res = api.predict(image=image, min_mask_score=0.9, bottom_crop=0)
X = np.zeros(res['image_shape'])
X[tuple(res['separation_coordinates'])] = 1
plt.clf(); plt.imshow(X); plt.show()
res['score']
