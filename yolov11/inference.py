import os
import sys
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sign_info_parser import get_sign_info, format_sign_info
from utils.label_const import LABEL_TEXT, LABEL_CHAR


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def detect_and_explain(model_path: str, image_path: str, threshold: float = 0.5, visualize: bool = True):
    """
    Detect signs using YOLO and print detailed information including penalties
    
    Args:
        model_path: Path to YOLO model (.pt file)
        image_path: Path to input image
        threshold: Confidence threshold
        visualize: Whether to show visualization
    """
    print(f"\n Đang phát hiện biển báo trong: {image_path}")
    print("=" * 60)
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path, conf=threshold)[0]
    
    if len(results.boxes) == 0:
        print(" Không phát hiện biển báo nào trong ảnh.")
        return
    
    # Extract detections
    detections = []
    for box in results.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        xyxy = box.xyxy[0].cpu().numpy()
        
        # YOLO uses 0-indexed classes, our labels are 1-indexed
        label_id = cls + 1
        
        detections.append({
            'box': xyxy,
            'label': label_id,
            'score': conf,
            'sign_code': LABEL_CHAR.get(label_id, 'Unknown'),
            'vietnamese_name': LABEL_TEXT.get(label_id, 'Unknown')
        })
    
    print(f"✅ Phát hiện {len(detections)} biển báo:\n")
    
    # Summary table
    print(f"{'#':<3} {'Mã biển':<15} {'Tên gọi':<35} {'Độ tin cậy':<10}")
    print("-" * 63)
    for i, det in enumerate(detections, 1):
        name = det['vietnamese_name'][:33] if len(det['vietnamese_name']) > 33 else det['vietnamese_name']
        print(f"{i:<3} {det['sign_code']:<15} {name:<35} {det['score']:.2%}")
    
    print("\n" + "=" * 60)
    print("📋 THÔNG TIN CHI TIẾT VÀ MỨC XỬ PHẠT:")
    print("=" * 60)
    
    # Detailed info for each unique detection
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
        # Use YOLO's built-in visualization
        annotated = results.plot()
        plt.figure(figsize=(14, 10))
        plt.imshow(annotated)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return detections


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect and explain traffic signs using YOLO")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="yolo-viet-signs/yolov8n_viet_signs/weights/best.pt", 
                        help="Path to YOLO model")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Confidence threshold")
    parser.add_argument("--no-viz", action="store_true", 
                        help="Disable visualization")
    
    args = parser.parse_args()
    
    # Check model path
    model_path = os.path.join(PROJECT_ROOT, "yolov11", args.model)
    if not os.path.exists(model_path):
        model_path = args.model
    
    if not os.path.exists(model_path):
        print(f" Không tìm thấy model: {model_path}")
        return
    
    # Run detection
    detect_and_explain(
        model_path, 
        args.image, 
        threshold=args.threshold,
        visualize=not args.no_viz
    )


if __name__ == "__main__":
    main()
