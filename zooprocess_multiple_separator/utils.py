# -*- coding: utf-8 -*-
"""
Functions used in api.py
(which allow to keep api.py simple)
  - prediction of panoptic masks
  - conversion into separating lines through a watershed algorithm
"""

import os
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms.v2 as tr
import torchvision.transforms.v2.functional as trf
from torchvision.io import read_image

from skimage.measure import label
from skimage.segmentation import watershed, find_boundaries


def prepare_zooscan_image(image, bottom_crop):
    """
    Transform the image input in the network
    
    This is meant to be used as the transform argument of a torch Dataset object
    
    Args:
        image (tensor): image tensor read by torchvision.io.read_image.
        bottom_crop (int): number of pixels to crop from the bottom of the image.
    
    Returns:
        The prepared tensor.
    """
    # (possibly) crop the bottom of the tensor
    d, h, w = image.size()
    image = trf.crop(image, 0, 0, h-bottom_crop, w)

    # preprocess the tensor
    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
    preprocess = tr.Compose([
        tr.Resize((512,512)),
        tr.ToDtype(torch.float32, scale=True),
        tr.Normalize(ADE_MEAN, ADE_STD)
    ])
    image = preprocess(image)
    
    return image


class ZooScanEvalDataset(Dataset):
    def __init__(self, paths, names, transform=None, bottom_crop=0):
      self.paths = paths
      self.names = names
      self.transform = transform
      self.bottom_crop = bottom_crop

    def __len__(self):
      return len(self.paths)

    def __getitem__(self, idx):
      # get the image name
      name = self.names[idx]
      # read the image
      image = read_image(self.paths[idx])
      # gets its dimensions
      d,h,w = image.size()
      dims = (h,w)
      # transform it, if need be
      if self.transform:
        image = self.transform(image, bottom_crop=self.bottom_crop)
      # TODO actually the transforms need to have resize at least

      # NB: we cannot return images that are not all the same size;
      #     it does not work in a dataloader afterwards:
      #     https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
      return image,name,dims


def predict_panoptic_masks(image_paths, image_names, model, processor, device, bottom_crop=31):
    """
    Perform the mask segmention for a given image with a panoptic model
    
    This runs the trained panoptic model, selects the masks above the score
    threshold, combines all masks and computes a distance map from the mask border,
    compute the center point of ach mask. Finally it returns variables relevant
    for the next part of the process.
    
    Args:
        image_paths (str): path to the image to process.
        model, processor: Mask2Former model and processor objects.
          Should be global variables generated by warm() here.
        device: CPU or CUDA device. Also generated by warm().
        min_mask_score (float): minimum probability to retain a potential mask.
        bottom_crop (int): number of pixels to crop from the bottom of the image
          (e.g. to remove the scale bar for example)
    
    Returns:
        panoptic_masks (np.ndarray): the labels of the retained masks.
        avg_score (float): average of the scores of the retained masks.
        image (Image): input image, possibly after crop.
    """
    
    ds = ZooScanEvalDataset(paths=image_paths, names=image_names,
                            transform=prepare_zooscan_image, bottom_crop=bottom_crop)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

    # set model in evaluation mode
    model.eval()
    torch.set_grad_enabled(False)

    # prepare storage of probabilities to be multiple (e.g. the score)
    # scores = []
    # iterate over batches
    inputs,images,names,dims=next(iter(dl))

    for images,inputs,names,dims in dl:
        # process batch through model, on GPU
        inputs = inputs.to(device)
        outputs = model(inputs)
        # convert from tensors to tuples of int
        dims = [(int(x[0]),int(x[1])) for x in dims]
        results = processor.post_process_panoptic_segmentation(outputs, target_sizes=dims)
        # panoptic_masks = 
        # # plt.clf(); plt.imshow(panoptic_masks); plt.show()
        
    # return outputs
  
    min_mask_score=0.9
    # 
    # # set model in evaluation mode
    # model.eval()
    # torch.set_grad_enabled(False)
    # 
    # # prepare storage of probabilities to be multiple (e.g. the score)
    # scores = []
    # # iterate over batches
    # for imgs,names in dl:
    #     # process batch through model, on GPU
    #     imgs = imgs.to(device)
    #     outputs = model(imgs)
    #     results = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    #     panoptic_masks = results["segmentation"].cpu().numpy()
    #     # plt.clf(); plt.imshow(panoptic_masks); plt.show()
    # 
    # 
    # img_tens = img_tens.to(device)      # copy to GPU/CPU
    # img_tens = img_tens[None, :, :, :]  # add empty dimension as for a batch
    # 
    # # predict panoptic masks
    # 
    
    for (i in range(results)):
      res = results[i]
      
      # keep only those with a high enough score
      selected_masks_ids = [seg["id"]  for seg in res["segments_info"]\
        if seg["label_id"] == 1 and seg["score"]>min_mask_score]
        # NB: label_id == 1 for objects, 0 for background
        
      # compute their average score, as an indication of the quality of the segmentation
      scores = [seg["score"] for seg in res["segments_info"]\
        if seg["id"] in selected_masks_ids]
      avg_score = np.mean(scores)
      # the scores are necessarily betweem min_mask_score and 1
      # rescale the average score between 0 and 1
      avg_score = (avg_score - min_mask_score) * 1 / (1-min_mask_score)

      # assign everything else as background (=0)
      panoptic_masks = res["segmentation"].cpu().numpy()
      panoptic_masks[~ np.isin(panoptic_masks, selected_masks_ids)] = 0
      # plt.clf(); plt.imshow(panoptic_masks); plt.show()

      # Now we need to detect large grey regions missed by the panoptic segmenter
      # and consider them as new masks, to improve the final segmentation

      # get binary image separating the background (0) from grey regions (1)
      # TODO stuck here because image ha sbeen read but is not avaiable to us; re-reading it sounds wasteful!
      gray_img = np.array(image.convert('L'))
      binary_img = (gray_img < 255).astype(float)
      # NB: using < 255 assumes the background is perfectly white
      # plt.clf(); plt.imshow(gray_img, cmap='Greys_r', interpolation='none'); plt.show()

      # detect missing regions = grey regions outside of masks detected by the panoptic segmenter
      missing_regions = np.logical_and(panoptic_masks == 0, binary_img != 0)
      # plt.clf(); plt.imshow(missing_regions); plt.show()
      missing_regions = label(missing_regions, background=0, return_num=False, connectivity=2)
      # plt.clf(); plt.imshow(missing_regions); plt.show()

      # keep only large missing regions and add them as new masks
      missing_regions_ids, nb_pixels = np.unique(missing_regions, return_counts=True)
      max_mask_id = int(np.max(selected_masks_ids))
      for i in np.delete(missing_regions_ids, 0):
          # NB: do not consider region 0 which is the background
          # if large enough, add it to the masks
          if nb_pixels[i]>800:
              # TODO make the threshold number of pixels (800 here) configurable
              max_mask_id = max_mask_id+1  # increase the mask id counter
              panoptic_masks[missing_regions == missing_regions_ids[i]] = max_mask_id
              selected_masks_ids.append(max_mask_id)
      # plt.clf(); plt.imshow(panoptic_masks); plt.show()

      # pad again with bottom crop (if not zero)
      # this allows to give a result that has exactly the same shape as the input
      if bottom_crop > 0:
          h,w = panoptic_masks.shape
          crop = np.zeros((bottom_crop, w))
          panoptic_masks = np.concatenate((panoptic_masks, crop), axis=0)
          binary_img = np.concatenate((binary_img, crop), axis=0)
          crop[:,:] = 255
          gray_img = np.concatenate((gray_img, crop), axis=0)
          # plt.clf(); plt.imshow(gray_img); plt.show()
          # plt.clf(); plt.imshow(binary_img); plt.show()
          # plt.clf(); plt.imshow(panoptic_masks); plt.show()

    # return panoptic_masks, avg_score, gray_img, binary_img


def separate_with_watershed(panoptic_masks, overall_mask=None):
    """
    Apply the watershed algorithm on the predicted mask map, using the mask
    centers as markers, to generate lines separating the different objects.
    
    Args:
        panoptic_masks (np.ndarray): array of masks labels output by
          `predict_panoptic_masks()`. 0 is the background, the other values are
          masks.
        overall_mask (np.ndarray): mask of the original image, over which to 
          compute the watershed. 0 is the background, 1 is the objects. If None,
          will be computed from the distance map computed by the watershed. This
          is also output by `predict_panoptic_masks()`.
      
    Returns:
        (np.ndarray) with 0 as background and 1 for lines separating two objects.
    """
    
    # compute distance map (distance to the edge of each mask) and the
    # coordinates of mask centers
    mask_ids = np.delete(np.unique(panoptic_masks), 0)
    # plt.clf(); plt.imshow(panoptic_masks); plt.show()
    dist_map = np.zeros(panoptic_masks.shape)
    center_x = list()
    center_y = list()
    for mask_id in mask_ids:
        single_mask = (panoptic_masks == mask_id).astype(int)
        # distance map
        dist = ndi.distance_transform_edt(single_mask)
        dist_map += dist
        # highest point, considered as the mask's center
        center_coords = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
        center_x.append(center_coords[0])
        center_y.append(center_coords[1])
    # plt.clf(); plt.imshow(dist_map); plt.show()
    # center_x
    # center_y

    # Prepare watershed markers
    watershed_markers = np.zeros(dist_map.shape, dtype=bool)
    watershed_markers[(center_x, center_y)] = True
    watershed_markers, _ = ndi.label(watershed_markers)
    # plt.clf(); plt.imshow(watershed_markers); plt.show()

    # Prepare watershed mask
    if overall_mask is None:
        watershed_mask = np.zeros(dist_map.shape, dtype='int64')
        watershed_mask[dist_map > .01] = 1
    else:
        watershed_mask = overall_mask
    # plt.clf(); plt.imshow(watershed_mask); plt.show()

    # Apply watershed
    labels = watershed(
        -dist_map, watershed_markers, mask=watershed_mask, watershed_line=False
    )
    # plt.clf(); plt.imshow(labels); plt.show()

    # Derive separation lines
    lines = np.zeros(labels.shape)
    unique_labels = list(np.unique(labels))
    unique_labels.remove(0)

    for value in unique_labels:
        single_shape = (labels == value).astype(int)
        boundaries = find_boundaries(
            single_shape, connectivity=2, mode='outer', background=0
        )
        boundaries[(labels == 0) | (labels == value)] = 0
        lines[boundaries == 1] = 1
    # plt.clf(); plt.imshow(lines); plt.show()
    
    return lines
