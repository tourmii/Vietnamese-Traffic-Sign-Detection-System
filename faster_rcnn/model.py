import torch 
import torch.nn as nn
import torchvision
import math 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_iou_matrix(boxes1, boxes2):
    r"""
        boxes1: (N, 4)
        boxes2: (M, 4)
        iou_matrix: (N, M)
    """
    # Area of boxes (x2 - x1) * (y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:,3] - boxes1[:,1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:,3] - boxes2[:,1])

    # get top left x1, y1
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])

    # get bottom right x2, y2
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union_area = area1[:, None] + area2 - intersection_area

    return intersection_area / union_area # (N, M)

def apply_regression_to_anchors(box_transform_pred, anchors_or_proposals):
    """
        box_transform_pred: (num_anchors_or_proposals, num_classes, 4)
        anchors_or_proposals: (num_anchors_or_proposals, 4)
        pred_bbox: (num_anchors_or_proposals, num_classes, 4)
    """
    box_transform_pred = box_transform_pred.reshape(
        box_transform_pred.size(0), -1, 4
    )

    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]

    center_x = anchors_or_proposals[:, 0] + 0.5 * w
    center_y = anchors_or_proposals[:, 1] + 0.5 * h

    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]

    # dh -> (num_anchors_or_proposals, num_classes)

    pred_center_x = dx * w[:, None] + center_x[:, None]
    pred_center_y = dy * h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]

    # pred_center_x -> (num_anchors_or_proposals, num_classes)
    pred_bbox_x1 = pred_center_x - 0.5 * pred_w
    pred_bbox_y1 = pred_center_y - 0.5 * pred_h
    pred_bbox_x2 = pred_center_x + 0.5 * pred_w
    pred_bbox_y2 = pred_center_y + 0.5 * pred_h

    pred_bbox = torch.stack(
        (pred_bbox_x1,
        pred_bbox_y1,
        pred_bbox_x2,
        pred_bbox_y2),
        dim=-2
    )

    return pred_bbox

def clamp_boxes_to_image_boundary(boxes, image_shape):
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]

    height, width = image_shape[-2:]

    boxes_x1 = boxes_x1.clamp(min=0, max=width)
    boxes_y1 = boxes_y1.clamp(min=0, max=height)
    boxes_x2 = boxes_x2.clamp(min=0, max=width)
    boxes_y2 = boxes_y2.clamp(min=0, max=height)

    boxes = torch.cat((
        boxes_x1[..., None],
        boxes_y1[..., None],
        boxes_x2[..., None],
        boxes_y2[..., None]
    ), dim=-1)
    return boxes

def boxes_to_transformation_targets(ground_truth_boxes, anchors):
    # Get center x, y, w, h from x1, y1, x2, y2 for anchors
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    center_x = anchors[:, 0] + 0.5 * widths 
    center_y = anchors[:, 1] + 0.5 * heights

    gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights

    target_dx = (gt_center_x - center_x) / widths
    target_dy = (gt_center_y - center_y) / heights
    target_dw = torch.log(gt_widths / widths)
    target_dh = torch.log(gt_heights / heights)

    regression_targets = torch.stack((
        target_dx,
        target_dy,
        target_dw,
        target_dh
    ), dim=-1)
    
    return regression_targets

def sample_positive_negative_anchors(labels, positive_count, total_count):
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]
    num_positive = min(positive_count, positive.numel())
    num_negative = total_count - num_positive 
    num_negative = min(num_negative, negative.numel())

    perm_positive_idx = torch.randperm(positive.numel(), device=labels.device)[:num_positive]
    perm_negative_idx = torch.randperm(negative.numel(), device=labels.device)[:num_negative]

    pos_idx = positive[perm_positive_idx]
    neg_idx = negative[perm_negative_idx] 

    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_pos_idx_mask[pos_idx] = True
    sampled_neg_idx_mask[neg_idx] = True

    return sampled_pos_idx_mask, sampled_neg_idx_mask

def transform_boxes_to_original_size(boxes, new_size, original_size):
    ratios = [
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device) / torch.tensor(s_new, dtype=torch.float32, device=boxes.device)
        for s_new, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    boxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
    return boxes        
 

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512):
        super(RegionProposalNetwork, self).__init__()
        self.scales = [128, 256, 512]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)

        # 3x3 conv 
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        #cls layer
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1, stride=1)

        #bbox regression layer
        self.bbox_reg_layer = nn.Conv2d(in_channels, self.num_anchors*4, kernel_size=1, stride=1)

    def filter_proposals(self, proposals, cls_score, image_shape):
        cls_score = cls_score.reshape(-1)
        cls_score = torch.sigmoid(cls_score)
        _, top_n_idx = cls_score.topk(10000)
        cls_score = cls_score[top_n_idx]
        proposals = proposals[top_n_idx]

        #clamp boxes to image boundary 
    
        proposals = self.clamp_boxes_to_image_boundary(proposals, image_shape)
        # NMS based on objectness 
        keep_mask = torch.zeros_like(cls_score, dtype=torch.bool)
        keep_indices = torch.ops.torchvision.nms(proposals, cls_score, 0.7)

        post_nms_top_n = keep_indices[cls_score[keep_indices].sort(descending=True)[1]]

        # Post NMS topk filtering 
        proposals = proposals[post_nms_top_n[:2000]]
        cls_score = cls_score[post_nms_top_n[:2000]]
        return proposals, cls_score

    def assign_targets_to_anchors(self, anchors, gt_boxes):
        # Get (gt_boxes, num_anchors) IoU matrix
        iou_matrix = get_iou_matrix(gt_boxes, anchors)

        # for each anchor get best gt box index
        best_gt_iou, best_gt_idx = iou_matrix.max(dim=0)

        # quality boxes
        best_match_gt_idx_pre_thres = best_gt_idx.clone()
        
        below_threshold = best_gt_iou < 0.3
        between_threshold = (best_gt_iou >= 0.3) & (best_gt_iou < 0.7)
        best_gt_idx[below_threshold] = -1
        best_gt_idx[between_threshold] = -2

        # low quality anchor boxes 
        best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1)
        gt_pred_pair_highest_iou = (iou_matrix == best_anchor_iou_for_gt[:, None])

        # get all anchor indices to update
        pred_indices_to_update = gt_pred_pair_highest_iou[1]
        best_gt_idx[pred_indices_to_update] = best_match_gt_idx_pre_thres[pred_indices_to_update] 

        # best match idx is either valid index or -1 (negative) or -2 (ignore) 
        matched_gt_boxes = gt_boxes[best_gt_idx.clamp(min=0)]

        labels = best_gt_idx >= 0
        labels = labels.to(dtype=torch.float32)

        backgroun_anchors = best_gt_idx == -1
        labels[backgroun_anchors] = 0.0

        ignore_anchors = best_gt_idx == -2
        labels[ignore_anchors] = -1.0

        return labels, matched_gt_boxes 

    def generate_anchors(self, image, feat):
        grid_h, grid_w = feat.shape[-2:]
        image_h, image_w = image.shape[-2:]

        stride_h = torch.tensor(image_h // grid_h, dtype = torch.int64, device= feat.device)

        stride_w = torch.tensor(image_w // grid_w, dtype = torch.int64, device=feat.device)

        scales = torch.tensor(self.scales, dtype=feat.type, device=feat.device)
        aspect_ratios = torch.tensor(self.aspect_ratios, type=feat.type, device=feat.device)

        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 /h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1)/2
        base_anchors = base_anchors.round()

        #get shifts in x axis (0, 1, ..., w_feat-1) * stride_w
        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device)*stride_w
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device)*stride_h

        shifts_x, shifts_y = torch.meshgrid(shifts_y, shifts_x, indexing='ij')


        #(H_feat, W_feat)
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)

        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)

        #shifts -> (H_feat * W_feat, 4)

        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))

        anchors = anchors.reshape(-1, 4)

        return anchors
    
    def forward(self, image, feat, target):
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))

        cls_score = self.cls_layer(rpn_feat)

        box_transform_pred = self.bbox_reg_layer(rpn_feat)

        anchors = self.generate_anchors(image, feat)

        #cls_score -> (N, A, H, W)

        num_anchors_per_loc = cls_score.size(1)
        cls_score = cls_score.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(-1, 1)
        

        box_transform_pred = box_transform_pred.view(
            box_transform_pred.size(0),
            num_anchors_per_loc,
            4,
            rpn_feat.shape[-2],
            rpn_feat.shape[-1]
        )

        box_transform_pred = box_transform_pred.permute(0, 3, 4, 1, 2)
        box_transform_pred = box_transform_pred.reshape(-1, 4)

        proposals = apply_regression_to_anchors(
            box_transform_pred.detach().reshape(-1, 1, 4), 
            anchors
        )
        proposals = proposals.reshape(proposals.size(0), 4)
        proposals, scores = self.filter_proposals(proposals, cls_score.detach(), image.shape)

        rpn_output = {
            'proposals': proposals,
            'scores': scores
        }

        if not self.training or target is None:
            return rpn_output
        else:
            labels_for_anchors, matched_gt_boxes = self.assign_targets_to_anchors(
                anchors,
                target['bboxes'][0]
            )

            # match_gt_boxes -> (num_anchors, 4)
            # anchors -> (num_anchors, 4)
            regression_targets = boxes_to_transformation_targets(
                matched_gt_boxes,
                anchors
            )

            # Sample positive and negative anchors for training
            sample_neg_idx_mask, sample_pos_idx_mask = sample_positive_negative_anchors(
                labels_for_anchors,
                positive_count=128,
                total_count=256
            )
            sample_idxs = torch.where(sample_pos_idx_mask | sample_neg_idx_mask)[0]

            localization_loss = (
                torch.nn.functional.smooth_l1_loss(
                    box_transform_pred[sample_pos_idx_mask],
                    regression_targets[sample_pos_idx_mask],
                    reduction='sum',
                    beta = 1/9,
                ) / (sample_idxs.numel())
            )

            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cls_score[sample_idxs].flatten(),
                labels_for_anchors[sample_idxs].flatten()
            )

            rpn_output['rpn_classification_loss'] = cls_loss
            rpn_output['rpn_regression_loss'] = localization_loss
            return rpn_output
        
class ROIHead(nn.Module):
    def __init__(self, num_classes=52, in_channels=512):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.pool_size = 7 
        self.fc_inner_dim = 1024

        self.fc6 = nn.Linear(in_channels*self.pool_size*self.pool_size, self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        self.cls_layer = nn.Linear(self.fc_inner_dim, num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, num_classes*4) 

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        iou_matrix = get_iou_matrix(gt_boxes, proposals)

        best_gt_iou, best_gt_idx = iou_matrix.max(dim=0)

        below_threshold = best_gt_iou < 0.5

        best_gt_idx[below_threshold] = -1

        matched_gt_boxes = gt_boxes[best_gt_idx.clamp(min=0)]

        labels = gt_labels[best_gt_idx.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)

        background_proposals = best_gt_idx == -1
        labels[background_proposals] = 0

        return labels, matched_gt_boxes

    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        
        keep = torch.where(pred_scores > 0.05)[0]
        pred_boxes, pred_labels, pred_scores = pred_boxes[keep], pred_labels[keep], pred_scores[keep]

        min_size = 1
        ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = torch.where((ws >= min_size) & (hs >= min_size))[0]
        pred_boxes, pred_labels, pred_scores = pred_boxes[keep], pred_labels[keep], pred_scores[keep]

        keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
        for class_id in torch.unique(pred_labels):
            curr_idx = torch.where(pred_labels == class_id)[0]
            curr_keep_idx = torch.ops.torchvision.nms(
                pred_boxes[curr_idx],
                pred_scores[curr_idx],
                0.5
            )
            keep_mask[curr_idx[curr_keep_idx]] = True

        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(descending=True)[1]]

        keep = post_nms_keep_indices[:100]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        return pred_boxes, pred_labels, pred_scores

    def forward(self, feat, proposals, image_shape, target=None):
        if self.training and target is not None:
            gt_boxes = target['bboxes'][0]
            gt_labels = target['labels'][0]

            labels, matched_gt_boxes = self.assign_targets_to_proposals(
                proposals,
                gt_boxes,
                gt_labels
            )

            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative_anchors(
                labels,
                positive_count=32,
                total_count=128
            )
            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
            proposals = proposals[sampled_idxs]
            labels = labels[sampled_idxs]
            matched_gt_boxes = matched_gt_boxes[sampled_idxs]

            regression_target = boxes_to_transformation_targets(
                matched_gt_boxes,
                proposals
            )
        
        #ROI Pooling
        #For vgg16, the feature map is 1/16 of the input image size
        spatial_scale = 1.0 / 16.0
        proposals_roi_pooling = torchvision.ops.roi_pool(
            feat,
            [proposals],
            output_size=self.pool_size,
            spatial_scale=spatial_scale
        )
        proposal_roi_pool_feats = proposals_roi_pooling.flatten(start_dim=1)
        box_fc6 = torch.nn.ReLU()(self.fc6(proposal_roi_pool_feats))
        box_fc7 = torch.nn.ReLU()(self.fc7(box_fc6))
        cls_score = self.cls_layer(box_fc7) 
        box_transform_pred = self.bbox_reg_layer(box_fc7)

        num_boxes, num_classes = cls_score.shape
        box_transform_output = box_transform_pred.reshape(num_boxes, num_classes, 4)

        frcnn_output = {}

        if self.training and target is not None:
            classification_loss = torch.nn.functional.cross_entropy(
                cls_score,
                labels
            )

            fg_proposals_idxs = torch.where(labels > 0)[0]

            fg_class_label = labels[fg_proposals_idxs]
            localization_loss = torch.nn.functional.smooth_l1_loss(
                box_transform_output[fg_proposals_idxs, fg_class_label],
                regression_target[fg_proposals_idxs],
                reduction='sum',
                beta=1/9
            ) / (labels.numel())

            frcnn_output['frcnn_classification_loss'] = classification_loss
            frcnn_output['frcnn_regression_loss'] = localization_loss
            return frcnn_output
        else:
            pred_boxes = apply_regression_to_anchors(
                box_transform_pred,
                proposals
            )
            pred_scores = torch.nn.functional.softmax(cls_score, dim=-1)

            # Clamp boxes to image boundary
            pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image_shape)

            # create label for each prediction
            pred_labels = torch.arange(num_classes, device=cls_score.device)
            pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)

            pred_boxes = pred_boxes[:, 1:]
            pred_scores = pred_scores[:, 1:]
            pred_labels = pred_labels[:, 1:]

            pred_boxes = pred_boxes.reshape(-1, 4)
            pred_scores = pred_scores.reshape(-1)
            pred_labels = pred_labels.reshape(-1)

            pred_boxes, pred_labels, pred_scores = self.filter_predictions(
                pred_boxes,
                pred_labels,    
                pred_scores
            )
            frcnn_output['boxes'] = pred_boxes
            frcnn_output['score'] = pred_scores 
            frcnn_output['labels'] = pred_labels

            return frcnn_output

class FasterRCNN(nn.Module):
    def __init__(self, num_classes=52):
        super(FasterRCNN, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.backbone = vgg16.features[:-1]  # Remove the last maxpool layer
        self.rpn = RegionProposalNetwork(in_channels=512)
        self.roi_head = ROIHead(num_classes=num_classes, in_channels=512)

        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = 600
        self.max_size = 1000
    
    def normalize_resize_image_and_boxes(self, image, bboxes):
        mean = torch.tensor(self.image_mean,
                            dtype=image.dtype,
                           device=image.device)
        
        std = torch.tensor(self.image_std,
                           dtype=image.dtype,
                           device=image.device)
        
        image = (image - mean[:, None, None]) / std[:, None, None]

        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(
            float(self.min_size) / min_size,
            float(self.max_size) / max_size
        )
        scale_factor = scale.item()


        image = torch.nn.functional.interpolate(
            image,
            size=None,
            scale_factor=scale_factor,
            model='bilinear',
            recompute_scale_factor=True,
            align_corners=False
        )

        if bboxes is not None:
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=bboxes.device) / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                for s, s_orig in zip(image.shape[-2:], (h, w))
            ]
            ratio_height, ratio_width = ratios
            xmin, ymin, xmax, ymax = bboxes.unbind(2)
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
        
        return image, bboxes

    def forward(self, image, target=None):
        old_shape = image.shape[-2:]
        if self.training:
            image, bboxes = self.normalize_resize_image_and_boxes(
                image,
                target['bboxes']
            )
            target['bboxes'] = bboxes
        else:
            image, _ = self.normalize_resize_image_and_boxes(
                image,
                None
            )

        feat = self.backbone(image)

        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']

        frcnn_output = self.roi_head(feat, proposals, image.shape[-2:])

        if not self.training:
            frcnn_output['boxes'] = transform_boxes_to_original_size(
                frcnn_output['boxes'],
                image.shape[-2:],
                old_shape
            )

        return rpn_output, frcnn_output
    