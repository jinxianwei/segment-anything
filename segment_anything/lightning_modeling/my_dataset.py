import os
from typing import Optional
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
# import torchvision.transforms as transforms
import torch
import albumentations as A
import torch
import numpy as np
from torch.nn import functional as F
from albumentations.pytorch import ToTensorV2
from skimage.measure import label, regionprops
from loguru import logger

def init_point_sampling(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
     # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels

def train_transforms(img_size, ori_h, ori_w):
    transforms = []
    if ori_h < img_size and ori_w < img_size:
        transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)

@logger.catch
def get_boxes_from_mask(mask, box_num=1, std = 0.1, max_pixel = 5):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
    label_img = label(mask)
    regions = regionprops(label_img)

    # Iterate through all regions and get the bounding box coordinates
    boxes = [tuple(region.bbox) for region in regions]

    # If the generated number of boxes is greater than the number of categories,
    # sort them by region area and select the top n regions
    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    # If the generated number of boxes is less than the number of categories,
    # duplicate the existing boxes
    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0,  y1, x1  = box
        width, height = abs(x1 - x0), abs(y1 - y0)
        # Calculate the standard deviation and maximum noise value
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
         # Add random noise to each coordinate
        noise_x = np.random.randint(-max_noise, max_noise)
        noise_y = np.random.randint(-max_noise, max_noise)
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))
    return torch.as_tensor(noise_boxes, dtype=torch.float)

class Segmentation_2D_Dataset():
    def __init__(self, 
                 mask_path='/home/bennie/bennie/bennie_project/segment-anything/ground-truth-pixel/', 
                 img_path='/home/bennie/bennie/bennie_project/segment-anything/scans/') -> None:
        self.mask_path_prefix = mask_path
        self.img_path_prefix = img_path
        self.all_ground_truth_masks = sorted(os.listdir(self.mask_path_prefix))[:100]

        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        self.image_size = 1024
    
    def __len__(self):
        return len(self.all_ground_truth_masks)
    
    def __getitem__(self, index):
        image_input = {}
        mask_name = self.all_ground_truth_masks[index]
        # print(mask_name)
        prefix = mask_name.split('.')[0][:-3]
        mask_path = os.path.join(self.mask_path_prefix, mask_name)
        gt_grayscale = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_grayscale == 0)
        gt_mask = gt_mask.astype(int)
        if gt_mask.max() == 255:
            gt_mask = gt_mask / 255
        
        img_path = os.path.join(self.img_path_prefix, prefix+'.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        img = (img - self.pixel_mean) / self.pixel_std
        
        h, w = gt_mask.shape
        gt_mask = np.array(gt_mask)
        transforms = train_transforms(self.image_size, h, w)
        augments = transforms(image=img, mask=gt_mask)
        
        image, mask = augments['image'].to(torch.float), augments['mask'].to(torch.int64)
        
        boxes = get_boxes_from_mask(mask)
        point_coords, point_labels = init_point_sampling(mask, 1)

        image_input['image'] = image
        image_input['label'] = mask.unsqueeze(0)
        image_input['point_coords'] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        
        return image_input  
        
if __name__ == "__main__":
    my_dataset = Segmentation_2D_Dataset()
    image_input = my_dataset.__getitem__(5)
    mask = image_input['label']
    boxes = image_input['boxes'].squeeze(0)
    image = image_input['image']
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.permute(1,2,0).numpy()
    boxes = np.array(boxes)
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots()
    ax.imshow(mask_np)
    rectangle = patches.Rectangle((boxes[0], boxes[1]), boxes[2]-boxes[0], boxes[3]-boxes[1], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rectangle)
    plt.savefig('mask_trans.png')
    
    plt.cla
    plt.clf
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    rectangle = patches.Rectangle((boxes[0], boxes[1]), boxes[2]-boxes[0], boxes[3]-boxes[1], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rectangle)
    plt.savefig("image_trans.png")

    
    
    
    
    