import numpy as np
import cv2
import time

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def inference_cuda(model_type: str = "vit_b", 
                   pth_path: str = "/home/bennie/bennie/bennie_project/segment-anything/demo/sam_vit_b_01ec64.pth", 
                   img_path: str = '/home/bennie/bennie/bennie_project/segment-anything/demo/20230116145226_500ns-00-ETD.tif', 
                   device: str = 'cuda',
                   resize_shape: tuple = (1050, 1050),
                   points_per_side: int = 32,
                   points_per_batch: int = 128,
                   mask_path: str = './my_mask.jpg'):
    t1 = time.time()
    # 读取图像并resize
    image = cv2.imread(img_path)
    ori_shape = image.shape[:2]
    image = cv2.resize(image, resize_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = pth_path

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)


    mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=points_per_side, points_per_batch=points_per_batch)

    masks = mask_generator.generate(image)

    result_mask = np.ones((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]))
    for i in range(1, len(masks)+1):
        m = masks[i-1]['segmentation'] * i 
        result_mask += m

    # 返回mask并resize
    result_mask = cv2.resize(result_mask, ori_shape)
    print(result_mask.shape)

    cv2.imwrite(mask_path, result_mask)  
    t2 = time.time()
    print(t2-t1)
    
    return result_mask, mask_path
    
if __name__ == "__main__":
    inference_cuda()
    
    
