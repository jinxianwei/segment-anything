import numpy as np
import cv2
import time
import torch
import onnxruntime

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

def inference_image(model_type: str = "vit_b", 
                   pth_path: str = "/home/bennie/bennie/bennie_project/segment-anything/demo/sam_vit_b_01ec64.pth", 
                   img_path: str = '/home/bennie/bennie/bennie_project/segment-anything/demo/20230116145226_500ns-00-ETD.tif', 
                   device: str = 'cuda',
                   resize_shape: tuple = (1050, 1050),
                   points_per_side: int = 32,
                   points_per_batch: int = 128,
                   mask_path: str = './my_mask.jpg'):
    """inference all instance in an image."""
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

def get_sam_model(model_type = 'vit_b', 
                  checkpoint_path=None, 
                  device='cuda:0'):
    checkpoint_path = '/home/bennie/bennie/bennie_project/segment-anything/demo/sam_vit_b_01ec64.pth'
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam_model.to(device)
    sam_model.eval()
    return sam_model

def image_process(img_path: str, sam_model, device):
    original_image = cv2.imread(img_path)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    input_image = transform.apply_image(image) # 改变尺寸为1024x1024
    input_image_torch = torch.as_tensor(input_image, device=device) # 转变为tensor
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    
    input_image = sam_model.preprocess(transformed_image) # 归一化
    original_image_size = image.shape[:2]
    input_size = tuple(transformed_image.shape[-2:])
    return input_image, input_size, original_image, original_image_size

def get_feature(input_image, sam_model):
    image_embedding = sam_model.image_encoder(input_image)
    return image_embedding

def resize_point(input_point, input_label, predictor, image):
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    return onnx_coord, onnx_label

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def after_upload():
    sam_model = get_sam_model()
    predictor = SamPredictor(sam_model)
    image = cv2.imread('/home/bennie/bennie/bennie_project/segment-anything/demo/20230116145226_500ns-00-ETD.tif')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 图片vit提特征
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    
    # 准备onnx模型
    onnx_model_path = "/home/bennie/bennie/bennie_project/segment-anything/demo/sam_onnx_quantized_example_b.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
    return image, image_embedding, ort_session, predictor

if __name__ == "__main__":
    # inference_image()
    import time
    t1 = time.time()
    # 交互式分割，需要传递图片提取特征
    image, image_embedding, ort_session, predictor = after_upload()
    t2  = time.time()
    print("image embedding:{}".format(t2-t1))
    # prompt的编码
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    onnx_coord, onnx_label = resize_point(input_point, input_label, predictor, image)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
}

    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold
    print(masks.shape)
    t3 = time.time()
    print("prompt and inference: {}".format(t3-t2))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.savefig("./result_0.jpg")
    t4 = time.time()
    print("save image{}".format(t4-t3))
    # plt.cla()
    # plt.clf()
    
    # prompt编码
    # input_point = np.array([[500, 375], [1125, 625]])
    # input_label = np.array([1, 1])
    # onnx_coord, onnx_label = prompt_emded(input_point, input_label, predictor, image, 'cuda:0')
    
    # 推理预测
    # TODO: 这里单独传入点的坐标吗，还是应该传入带有label的格式的点呢
    # sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
    #                                                 points=(onnx_coord, onnx_label),
    #                                                 boxes=None,
    #                                                 masks=None,
    #                                             )
    # low_res_masks, iou_predictions = sam_model.mask_decoder(
    #                                         image_embeddings=image_embedding,
    #                                         image_pe=sam_model.prompt_encoder.get_dense_pe(),
    #                                         sparse_prompt_embeddings=sparse_embeddings,
    #                                         dense_prompt_embeddings=dense_embeddings,
    #                                         multimask_output=False,
    #                                         )
    # upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to('cuda:0')
    
    
    
    
    
    
    
    
