import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def extract_feature_process(image_path,
                            resize_shape,
                            checkpoint_path,
                            model_type,
                            device):
    t0 = time.time()
    image = cv2.imread(image_path)
    image = cv2.resize(image, resize_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    predictor.set_image(image)
    
    return predictor

def embed_point_predict(input_point=None,
                        input_label=None,
                        predictor=None, 
                        mask_input=None):
    # input_point = np.array([[500, 375]])
    # input_label = np.array([1])
    if mask_input is not None:
        mask_input = mask_input[None, :, :]
        
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input,
        multimask_output=True)
    max_score_idx = np.argmax(scores)
    mask_input = logits[max_score_idx, :, :]
    mask = masks[max_score_idx].astype(int)
    return mask, mask_input
        
def embed_box_and_point_predict(input_point=None,
                                input_label=None,
                                input_box=None,
                                predictor=None):

    masks, _, _ = predictor.predict_torch(
        point_coords=input_point,
        point_labels=input_label,
        boxes=input_box,
        multimask_output=False)
    return masks

if __name__ == "__main__":
    # for visualize
    image = cv2.imread('./demo/20230116145226_500ns-00-ETD.tif')
    image = cv2.resize(image, (1050 ,1050))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictor = extract_feature_process(image_path='./demo/20230116145226_500ns-00-ETD.tif',
                                        resize_shape=((1050 ,1050)),
                                        checkpoint_path='./demo/sam_vit_b_01ec64.pth',
                                        model_type='vit_b',
                                        device='cuda')
    

    
    input_boxes = torch.tensor([
    [900, 900, 1050, 1050],
    [0, 0, 400, 400],
    [0, 800, 150, 1050]], device=predictor.device)
    input_point = torch.tensor(np.array([[20, 20]]), device=predictor.device)
    input_label = torch.tensor(np.array([0]), device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    transformed_points = predictor.transform.apply_coords_torch(input_point, image.shape[:2])
    masks  = embed_box_and_point_predict(input_point=transformed_points,
                                         input_label=input_label,
                                         input_box=transformed_boxes,
                                         predictor=predictor)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.savefig("./mask3.png")
    
    
    
    # input_point = np.array([[500, 375]])
    # input_label = np.array([1])
    # mask, mask_input = embed_point_predict(input_point=input_point,
    #                                        input_label=input_label,
    #                                        predictor=predictor)
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_mask(mask, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('off')
    # plt.savefig("./mask0.png")  
    
    # input_point = np.array([[500, 375], [20, 20]])
    # input_label = np.array([1, 1])
    # mask, mask_input = embed_point_predict(input_point=input_point,
    #                                        input_label=input_label,
    #                                        predictor=predictor,
    #                                        mask_input=mask_input)
    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_mask(mask, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('off')
    # plt.savefig("./mask1.png")  
    
    
    # input_point = np.array([[500, 375], [20, 20]])
    # input_label = np.array([1, 0])
    # mask, mask_input = embed_point_predict(input_point=input_point,
    #                                        input_label=input_label,
    #                                        predictor=predictor,
    #                                        mask_input=mask_input)
    # print(mask.shape)
    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_mask(mask, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('off')
    # plt.savefig("./mask2.png")  
    