import os
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

import torch
import torchvision
import torchvision.transforms.v2 as tr
import torchvision.transforms.v2.functional as trf

from skimage.measure import label
from skimage.segmentation import watershed, find_boundaries


def predict_mask_panoptic(image_path, model, processor, device, score_threshold=0.9, bottom_crop=31):
    """
    Perform the mask segmention for a given image with a panoptic model
    """
    img = Image.open(image_path)

    # (possibly) crop the bottom of the image
    w, h = img.size
    img = trf.crop(img, 0, 0, h-bottom_crop, w)

    # create and preprocess the tensor
    img_tens = trf.to_image(img.convert("RGB"))
    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
    preprocess = tr.Compose([
        tr.Resize((512,512)),
        tr.ToDtype(torch.float32, scale=True),
        tr.Normalize(ADE_MEAN, ADE_STD)
    ])
    img_tens = preprocess(img_tens)
    img_tens = img_tens.to(device)      # copy to GPU/CPU
    img_tens = img_tens[None, :, :, :]  # add empty dimension as for a batch
    
    # predict panoptic masks
    with torch.no_grad():
        outputs = model(img_tens)
    results = processor.post_process_panoptic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]
    panoptic_masks = results["segmentation"].cpu().numpy()
   
    # get binary image separating the background (0) from objects (1)
    # this assumes the background is perfectly white
    gray_img = np.array(img.convert('L'))
    binary_image = (gray_img < 255).astype(float)

    # Compute distances and mask centers
    distances = np.zeros(panoptic_masks.shape)
    mask_centers = list()
    scores = list()

    list_segments_obj_detectes = [i for i,d in enumerate(results["segments_info"])\
      if results["segments_info"][i]["label_id"] ==1 and results["segments_info"][i]["score"]>0.9]

    for segment_info in results["segments_info"]:
        if segment_info["score"] < score_threshold:
            continue
        if segment_info["label_id"] == 0:
            background_img = np.full(gray_img.shape,len(list_segments_obj_detectes)+1)
            for i in list_segments_obj_detectes:
                background_img[panoptic_masks == results["segments_info"][i]["id"]] = 0
            background_img[binary_image == 0] = 0
            obj_non_detect = label(background_img, background=0, return_num=False, connectivity=2)
            cpt_non_detect=0
            for label_value in np.unique(obj_non_detect):
                if label_value == 0:  # ignore background
                    continue
                if (obj_non_detect == label_value).astype(int).sum() > 800:  # very small regions are considered as background
                    single_mask = (obj_non_detect == label_value).astype(int)
                    dist = ndi.distance_transform_edt(single_mask)
                    distances += dist
                    ind = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
                    mask_centers.append((ind[0], ind[1]))
                    scores.append(segment_info["score"])
                    cpt_non_detect+=1
        else:
            single_mask = (panoptic_masks == segment_info["id"]).astype(int)
            dist = ndi.distance_transform_edt(single_mask)
            distances += dist
            ind = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
            mask_centers.append((ind[0], ind[1]))
            scores.append(segment_info["score"])

    return distances, mask_centers, img, binary_image, np.mean(scores)


def get_watershed_result(mask_map, mask_centers, mask=None):
    """
    Apply the watershed algorithm on the predicted mask map,
    using the mask centers as markers
    """
    # Prepare watershed markers
    markers_mask = np.zeros(mask_map.shape, dtype=bool)
    for (x, y) in mask_centers:
        markers_mask[x, y] = True
    markers, _ = ndi.label(markers_mask)

    # Prepare watershed mask
    if mask is None:
        watershed_mask = np.zeros(mask_map.shape, dtype='int64')
        watershed_mask[mask_map > .01] = 1
    else:
        watershed_mask = mask

    # Apply watershed
    labels = watershed(
        -mask_map, markers, mask=watershed_mask, watershed_line=False
    )

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

    labels_with_lines = labels
    labels_with_lines[labels_with_lines == 0] = -1
    labels_with_lines[lines == 1] = 0

    return labels_with_lines
