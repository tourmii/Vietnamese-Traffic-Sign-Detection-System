FROM python:3.10-slim

# System deps (OpenCV needs libGL + libglib)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Python dependencies ----------
COPY demo/requirements.txt /app/demo/requirements.txt
RUN pip install --no-cache-dir -r /app/demo/requirements.txt python-dotenv mistralai

# ---------- Model weights ----------
# YOLO weights
COPY yolov11/yolo-viet-signs/yolov8n_viet_signs2/weights/best.pt \
     /app/yolov11/yolo-viet-signs/yolov8n_viet_signs2/weights/best.pt

# Faster R-CNN weights
COPY faster_rcnn/checkpoints/best_model_1.pth \
     /app/faster_rcnn/checkpoints/best_model_1.pth

# ---------- Utility / knowledge files ----------
COPY utils/classes.txt        /app/utils/classes.txt
COPY utils/classes_vie.txt    /app/utils/classes_vie.txt
COPY utils/label_const.py     /app/utils/label_const.py
COPY utils/sign_info_parser.py /app/utils/sign_info_parser.py

COPY text_data/ /app/text_data/

# ---------- Demo application ----------
COPY demo/app.py      /app/demo/app.py
COPY demo/chatbot.py  /app/demo/chatbot.py
COPY demo/detector.py /app/demo/detector.py
COPY demo/.env        /app/demo/.env
# ---------- Runtime ----------
EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "demo.app"]