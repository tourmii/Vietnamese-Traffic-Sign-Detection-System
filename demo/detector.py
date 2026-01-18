import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
from typing import List, Dict, Tuple, Optional

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sign_info_parser import get_sign_info, format_sign_info
from utils.label_const import LABEL_TEXT, LABEL_CHAR, NUM_CLASSES


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TrafficSignDetector:
    """Unified detector supporting both Faster R-CNN and YOLO models."""
    
    def __init__(self):
        self.faster_rcnn_model = None
        self.yolo_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Default model paths
        self.faster_rcnn_path = os.path.join(
            PROJECT_ROOT, "faster_rcnn/checkpoints/best_model_1.pth"
        )
        self.yolo_path = os.path.join(
            PROJECT_ROOT, "yolov11/yolo-viet-signs/yolov8n_viet_signs2/weights/best.pt"
        )
    
    def _load_faster_rcnn(self):
        """Load Faster R-CNN model lazily."""
        if self.faster_rcnn_model is None:
            import torchvision
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            
            model = fasterrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)
            
            if os.path.exists(self.faster_rcnn_path):
                checkpoint = torch.load(self.faster_rcnn_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f" Loaded Faster R-CNN from {self.faster_rcnn_path}")
            else:
                print(f" Faster R-CNN checkpoint not found: {self.faster_rcnn_path}")
            
            model.to(self.device)
            model.eval()
            self.faster_rcnn_model = model
        
        return self.faster_rcnn_model
    
    def _load_yolo(self):
        """Load YOLO model lazily."""
        if self.yolo_model is None:
            from ultralytics import YOLO
            
            if os.path.exists(self.yolo_path):
                self.yolo_model = YOLO(self.yolo_path)
                print(f"✅ Loaded YOLO from {self.yolo_path}")
            else:
                print(f"⚠️ YOLO checkpoint not found: {self.yolo_path}")
                self.yolo_model = YOLO("yolov8n.pt")  # Fallback to pretrained
        
        return self.yolo_model
    
    def _apply_nms(self, boxes, scores, labels, iou_threshold=0.5):
        """Apply Non-Maximum Suppression."""
        import torchvision
        
        if len(boxes) == 0:
            return [], [], []
        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
        keep_indices = keep_indices.numpy()
        
        return boxes[keep_indices], scores[keep_indices], labels[keep_indices]
    
    def _detect_faster_rcnn(self, image: np.ndarray, threshold: float = 0.5, 
                            nms_threshold: float = 0.3) -> List[Dict]:
        """Detect using Faster R-CNN model."""
        import torchvision
        
        model = self._load_faster_rcnn()
        
        # Convert to tensor
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = torchvision.transforms.functional.to_tensor(image_pil)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(image_tensor)[0]
        
        # Filter by threshold
        keep = predictions['scores'] > threshold
        boxes = predictions['boxes'][keep].cpu().numpy()
        labels = predictions['labels'][keep].cpu().numpy()
        scores = predictions['scores'][keep].cpu().numpy()
        
        # Apply NMS
        boxes, scores, labels = self._apply_nms(boxes, scores, labels, nms_threshold)
        
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            detections.append({
                'box': box.tolist(),
                'label': int(label),
                'score': float(score),
                'sign_code': LABEL_CHAR.get(int(label), 'Unknown'),
                'vietnamese_name': LABEL_TEXT.get(int(label), 'Unknown')
            })
        
        return detections
    
    def _detect_yolo(self, image: np.ndarray, threshold: float = 0.5,
                     nms_threshold: float = 0.3) -> List[Dict]:
        """Detect using YOLO model."""
        model = self._load_yolo()
        
        # Run inference
        results = model(image, conf=threshold, iou=nms_threshold, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].cpu().numpy()
            
            label_id = cls + 1  # YOLO is 0-indexed
            
            detections.append({
                'box': xyxy.tolist(),
                'label': label_id,
                'score': conf,
                'sign_code': LABEL_CHAR.get(label_id, 'Unknown'),
                'vietnamese_name': LABEL_TEXT.get(label_id, 'Unknown')
            })
        
        return detections
    
    def detect_image(self, image_path: str, model_type: str = "yolo",
                     threshold: float = 0.5, nms_threshold: float = 0.3) -> Tuple[np.ndarray, List[Dict], str]:
        """
        Detect traffic signs in an image.
        
        Args:
            image_path: Path to input image
            model_type: "faster_rcnn" or "yolo"
            threshold: Confidence threshold
            nms_threshold: NMS threshold
            
        Returns:
            Tuple of (annotated_image, detections, info_text)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect
        if model_type.lower() == "faster_rcnn":
            detections = self._detect_faster_rcnn(image, threshold, nms_threshold)
        else:
            detections = self._detect_yolo(image, threshold, nms_threshold)
        
        # Annotate image
        annotated = self._annotate_image(image.copy(), detections)
        
        # Generate info text
        info_text = self._format_detections(detections)
        
        return annotated, detections, info_text
    
    def detect_video(self, video_path: str, model_type: str = "yolo",
                     threshold: float = 0.5, nms_threshold: float = 0.3,
                     process_every_n: int = 3) -> Tuple[str, List[Dict], str]:
        """
        Detect traffic signs in a video.
        
        Args:
            video_path: Path to input video
            model_type: "faster_rcnn" or "yolo"
            threshold: Confidence threshold
            nms_threshold: NMS threshold
            process_every_n: Process every N frames for speed
            
        Returns:
            Tuple of (output_video_path, all_detections, info_text)
        """
        import imageio
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # Output video path
        output_path = tempfile.mktemp(suffix='.mp4')
        
        all_detections = []
        unique_signs = set()
        frame_idx = 0
        last_detections = []
        frames = []
        last_annotated = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every N frames for speed
            if frame_idx % process_every_n == 0:
                if model_type.lower() == "faster_rcnn":
                    last_detections = self._detect_faster_rcnn(frame, threshold, nms_threshold)
                else:
                    last_detections = self._detect_yolo(frame, threshold, nms_threshold)
                
                # Track unique signs
                for det in last_detections:
                    unique_signs.add((det['label'], det['sign_code'], det['vietnamese_name']))
            
            # Always annotate with last detections (with side panel for video)
            annotated = self._annotate_image(frame.copy(), last_detections, add_side_panel=True)
            
            # Convert BGR to RGB for imageio
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frames.append(annotated_rgb)
            last_annotated = annotated
            
            all_detections.extend(last_detections)
            frame_idx += 1
        
        cap.release()
        
        # Write video using imageio
        try:
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', 
                                       pixelformat='yuv420p', 
                                       output_params=['-crf', '23'])
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        except Exception as e:
            print(f"Video writing failed: {e}, trying fallback...")
            try:
                # Fallback: try without specific codec
                imageio.mimwrite(output_path, frames, fps=fps)
            except Exception as e2:
                print(f"Fallback also failed: {e2}, saving as image...")
                # Last resort: save last frame as image
                output_path = tempfile.mktemp(suffix='.jpg')
                if last_annotated is not None:
                    cv2.imwrite(output_path, last_annotated)
        
        # Convert unique signs to detection format
        unique_detections = [
            {'label': label, 'sign_code': code, 'vietnamese_name': name, 'score': 1.0, 'box': []}
            for label, code, name in unique_signs
        ]
        
        # Generate info text for unique signs
        info_text = self._format_detections(unique_detections)
        
        return output_path, unique_detections, info_text
    
    def _annotate_image(self, image: np.ndarray, detections: List[Dict], 
                         add_side_panel: bool = False) -> np.ndarray:
        """Draw bounding boxes and labels on image with optional side panel."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        
        for i, det in enumerate(detections):
            box = det['box']
            if len(box) < 4:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            color = colors[det['label'] % len(colors)]
            
            # Draw box with thicker border
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label = f"{det['sign_code']}: {det['score']:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add side panel with sign info for video
        if add_side_panel:
            h, w = image.shape[:2]
            panel_width = 350
            
            # Create side panel (dark background)
            panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
            panel[:] = (30, 30, 40)  # Dark gray-blue
            
            # Add title
            cv2.putText(panel, "DETECTED SIGNS", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.line(panel, (10, 50), (panel_width - 10, 50), (100, 100, 100), 2)
            
            if detections:
                # List unique signs
                seen_labels = set()
                y_offset = 80
                
                for det in detections:
                    if det['label'] in seen_labels:
                        continue
                    seen_labels.add(det['label'])
                    
                    color = colors[det['label'] % len(colors)]
                    
                    # Draw colored indicator
                    cv2.circle(panel, (25, y_offset), 10, color, -1)
                    
                    # Sign code
                    cv2.putText(panel, det['sign_code'], (45, y_offset + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Vietnamese name (truncated)
                    name = det['vietnamese_name']
                    if len(name) > 25:
                        name = name[:22] + "..."
                    cv2.putText(panel, name, (45, y_offset + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                    
                    # Confidence
                    conf_text = f"Conf: {det['score']:.1%}"
                    cv2.putText(panel, conf_text, (45, y_offset + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 100), 1)
                    
                    y_offset += 75
                    
                    # Stop if panel is full
                    if y_offset > h - 50:
                        break
            else:
                # No detections message
                cv2.putText(panel, "No signs detected", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            # Combine image and panel
            image = np.hstack([image, panel])
        
        return image
    
    def _format_detections(self, detections: List[Dict]) -> str:
        """Format detection results with sign information (no penalties)."""
        if not detections:
            return " Không phát hiện biển báo nào trong ảnh."
        
        lines = []
        lines.append(f" **Phát hiện {len(detections)} biển báo:**\n")
        
        # Summary table
        lines.append("| # | Mã biển | Tên gọi | Độ tin cậy |")
        lines.append("|---|---------|---------|------------|")
        
        for i, det in enumerate(detections, 1):
            name = det['vietnamese_name'][:30] + "..." if len(det['vietnamese_name']) > 30 else det['vietnamese_name']
            score = f"{det['score']:.1%}" if det['score'] > 0 else "N/A"
            lines.append(f"| {i} | {det['sign_code']} | {name} | {score} |")
        
        lines.append("\n---\n")
        lines.append("## Thông tin chi tiết:\n")
        
        # Detailed info (without penalty)
        seen_labels = set()
        for det in detections:
            label = det['label']
            if label in seen_labels:
                continue
            seen_labels.add(label)
            
            info = get_sign_info(label)
            if 'error' not in info:
                lines.append(f"###  {info['sign_code']} - {info['vietnamese_name']}")
                lines.append(f"**Phân loại:** {info['category']}\n")
                
                if info.get('regulation_info'):
                    lines.append("**Quy định (QCVN 41:2019/BGTVT):**")
                    # Limit text length
                    reg_text = info['regulation_info'][:500]
                    if len(info['regulation_info']) > 500:
                        reg_text += "..."
                    lines.append(f"> {reg_text}\n")
                
                lines.append("\n---\n")
        
        lines.append("\n **Hỏi chatbot bên dưới để biết thêm về mức xử phạt và quy định liên quan!**")
        
        return "\n".join(lines)
    
    def get_detected_signs_context(self, detections: List[Dict]) -> str:
        """Get context string for chatbot about detected signs."""
        if not detections:
            return ""
        
        context_parts = []
        seen_labels = set()
        
        for det in detections:
            label = det['label']
            if label in seen_labels:
                continue
            seen_labels.add(label)
            
            info = get_sign_info(label)
            if 'error' not in info:
                part = f"- Biển {info['sign_code']}: {info['vietnamese_name']} ({info['category']})"
                if info.get('penalty_info'):
                    part += f"\n  Mức phạt: {info['penalty_info'][:200]}..."
                context_parts.append(part)
        
        return "\n".join(context_parts)


# Singleton instance
detector = TrafficSignDetector()


def detect_image(image_path: str, model_type: str = "yolo",
                 threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict], str]:
    """Convenience function for image detection."""
    return detector.detect_image(image_path, model_type, threshold)


def detect_video(video_path: str, model_type: str = "yolo",
                 threshold: float = 0.5) -> Tuple[str, List[Dict], str]:
    """Convenience function for video detection."""
    return detector.detect_video(video_path, model_type, threshold)
