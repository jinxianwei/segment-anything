from typing import Any
import random
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from metrics import SegMetrics
import numpy as np

from segment_anything import sam_model_registry


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss

class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss

class FocalDiceloss_IoULoss(nn.Module):
    
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss

class Segmentation_2d(pl.LightningModule):
    def __init__(self, 
                 model_type: str = 'vit_b',
                 checkpoint_path: str = 'sam_vit_b_01ec64.pth',
                 lr: float = 1e-4) -> None:
        super().__init__()
        self.lr = lr
        self.model = sam_model_registry[model_type](checkpoint_path)
        self.loss_fn = FocalDiceloss_IoULoss()
        
        self.epoch_iou = []
        self.epoch_dice = []
        
        # 冻结两者
        for name, param in self.model.named_parameters():
            if name.startswith("image_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)
        
        
    def forward(self, input):
        input_image = input['image']
        # with torch.no_grad(): # debug过程中，实际上通过了这个上下文，参数已冻结，pytorch_lightning开始时的参数总结是来自哪里呢
        labels = input['label']
        image_embeddings = self.model.image_encoder(input_image)
        batch, _, _, _ = image_embeddings.shape
        
        
        # 以随机某一种prompt进行训练
        if random.random() > 0.5:
            input["point_coords"] = None
            points = None
        else:
            input["boxes"] = None
            points = (input["point_coords"], input["point_labels"])
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=input.get("boxes", None),
            masks=input.get("mask_inputs", None),
        )
        low_res_masks, iou_predictions = self.model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = self.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
        masks = F.interpolate(low_res_masks,(1024, 1024), mode="bilinear", align_corners=False,)  # 这里的尺寸还原为多少，应该和mask的大小一致吧
        return masks, low_res_masks, iou_predictions

    
    def training_step(self, batch, batch_idx):
        gt_mask = batch['label']
        masks, low_res_masks, iou_predictions = self(batch)
        
        loss = self.loss_fn(masks, gt_mask, iou_predictions)
        self.log('train_step_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        gt_mask = batch['label']
        masks, low_res_masks, iou_predictions = self(batch)
        train_batch_metrics = SegMetrics(masks, gt_mask, ['iou', 'dice'])
        self.epoch_iou.append(train_batch_metrics[0])
        self.epoch_dice.append(train_batch_metrics[1])
        return train_batch_metrics
    
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            mean_epoch_iou = np.mean(self.epoch_iou)
            mean_epoch_dice = np.mean(self.epoch_dice)
            self.log("test_iou", mean_epoch_iou)
            self.log("test_dice", mean_epoch_dice)
        self.epoch_dice = []
        self.epoch_iou = []
    
    def configure_optimizers(self) -> Any:
        self.optimizer = optim.SGD(params=self.model.mask_decoder.parameters(),
                                   lr=self.lr)
        return self.optimizer