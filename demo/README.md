# 🚸 Vietnamese Traffic Sign Detection Demo

A web-based demo application for Vietnamese traffic sign detection with AI chatbot support.

## Features

- **Image Detection**: Upload images to detect traffic signs
- **Video Detection**: Process videos for continuous sign detection
- **Model Selection**: Choose between YOLO or Faster R-CNN
- **Sign Information**: View detailed regulations and penalty information
- **AI Chatbot**: Ask questions about traffic signs and Vietnamese traffic law

## Requirements

- Python 3.9+
- CUDA (optional, for GPU acceleration)
- Gemini API key (for chatbot feature)

## Installation

1. Install dependencies:
```bash
cd demo
pip install -r requirements.txt
```

2. Set up Gemini API key (for chatbot):
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## Usage

### Run the Demo

```bash
cd /path/to/Vietnamese-Traffic-Sign-Detection-System
python demo/app.py
```

Then open http://localhost:7860 in your browser.

### Model Files

Ensure model files are present at:
- **Faster R-CNN**: `faster_rcnn/checkpoints/best_model.pth`
- **YOLO**: `yolov11/yolo-viet-signs/yolov8n_viet_signs/weights/best.pt`

## Interface

### 📷 Image Detection Tab
1. Upload an image
2. Select model type (YOLO or Faster R-CNN)
3. Adjust confidence threshold
4. Click "Phát hiện biển báo" to detect

### 🎬 Video Detection Tab
1. Upload a video
2. Select model type
3. Adjust confidence threshold
4. Click "Phát hiện biển báo" to process

### 💬 Chatbot Tab
Ask questions about:
- Traffic sign meanings
- Traffic regulations (QCVN 41:2019/BGTVT)
- Penalty information (Nghị định 168/2024/NĐ-CP)
- Situational driving questions

Example questions:
- "Biển P.102 có ý nghĩa gì?"
- "Đi ngược chiều bị phạt bao nhiêu?"
- "Khi nào được quay đầu xe?"

## File Structure

```
demo/
├── app.py           # Main Gradio application
├── detector.py      # Unified detection module
├── chatbot.py       # Gemini chatbot integration
├── requirements.txt # Dependencies
└── README.md        # This file
```

## License

Part of Vietnamese Traffic Sign Detection System.
