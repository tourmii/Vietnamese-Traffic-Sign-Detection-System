import os
import sys
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sign_info_parser import get_sign_info, format_sign_info, load_classes
from utils.label_const import LABEL_TEXT, LABEL_CHAR, NUM_CLASSES


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(checkpoint_path: str, num_classes: int = 53):
    """Load trained Faster R-CNN model"""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def apply_nms(boxes, scores, labels, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove duplicate boxes"""
    if len(boxes) == 0:
        return [], [], []
    
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    
    # Apply NMS
    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
    keep_indices = keep_indices.numpy()
    
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]


def detect_signs(model, image_path: str, device: str = 'cuda', threshold: float = 0.5, nms_threshold: float = 0.3):
    """
    Detect traffic signs in an image
    
    Args:
        model: Trained Faster R-CNN model
        image_path: Path to the image
        device: 'cuda' or 'cpu'
        threshold: Confidence threshold
        nms_threshold: IoU threshold for NMS (lower = more aggressive filtering)
        
    Returns:
        List of detections with boxes, labels, scores
    """
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    # Filter by threshold
    keep = predictions['scores'] > threshold
    boxes = predictions['boxes'][keep].cpu().numpy()
    labels = predictions['labels'][keep].cpu().numpy()
    scores = predictions['scores'][keep].cpu().numpy()
    
    # Apply NMS to remove duplicate boxes
    boxes, scores, labels = apply_nms(boxes, scores, labels, nms_threshold)
    
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        detections.append({
            'box': box,
            'label': int(label),
            'score': float(score),
            'sign_code': LABEL_CHAR.get(int(label), 'Unknown'),
            'vietnamese_name': LABEL_TEXT.get(int(label), 'Unknown')
        })
    
    return detections, image


def visualize_detections(image, detections, save_path: str = None):
    """Visualize detections on the image"""
    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(image)
    
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    for i, det in enumerate(detections):
        box = det['box']
        label = det['label']
        score = det['score']
        name = det['vietnamese_name']
        code = det['sign_code']
        
        color = colors[label % 20]
        
        # Draw box
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        label_text = f"{code}: {score:.2f}"
        ax.text(
            box[0], box[1] - 5,
            label_text,
            fontsize=10, color='white',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def detect_and_explain(model, image_path: str, device: str = 'cuda', 
                       threshold: float = 0.5, visualize: bool = True):
    print(f"\n Đang phát hiện biển báo trong: {image_path}")
    print("=" * 60)
    
    detections, image = detect_signs(model, image_path, device, threshold)
    
    if not detections:
        print(" Không phát hiện biển báo nào trong ảnh.")
        return
    
    print(f" Phát hiện {len(detections)} biển báo:\n")
    
    # Summary table
    print(f"{'#':<3} {'Mã biển':<15} {'Tên gọi':<35} {'Độ tin cậy':<10}")
    print("-" * 63)
    for i, det in enumerate(detections, 1):
        print(f"{i:<3} {det['sign_code']:<15} {det['vietnamese_name'][:33]:<35} {det['score']:.2%}")
    
    print("\n" + "=" * 60)
    print(" THÔNG TIN CHI TIẾT VÀ MỨC XỬ PHẠT:")
    print("=" * 60)
    
    # Detailed info for each detection
    seen_labels = set()
    for det in detections:
        label = det['label']
        if label in seen_labels:
            continue
        seen_labels.add(label)
        
        info = get_sign_info(label)
        formatted = format_sign_info(info)
        print(formatted)
    
    # Visualize if requested
    if visualize:
        visualize_detections(image, detections, save_path="output.png")
    
    return detections


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect and explain traffic signs")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", 
                        help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.7, 
                        help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--no-viz", action="store_true", 
                        help="Disable visualization")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    model_path = os.path.join(PROJECT_ROOT, "faster_rcnn", args.model)
    if not os.path.exists(model_path):
        model_path = args.model
    
    if not os.path.exists(model_path):
        print(f" Không tìm thấy model: {model_path}")
        print("Hãy train model trước bằng: python train_faster_rcnn.py")
        return
    
    # Load model
    print(f" Đang tải model từ: {model_path}")
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, NUM_CLASSES + 1)  
    
    # Run detection
    detect_and_explain(
        model, 
        args.image, 
        device=device, 
        threshold=args.threshold,
        visualize=not args.no_viz
    )


if __name__ == "__main__":
    main()
