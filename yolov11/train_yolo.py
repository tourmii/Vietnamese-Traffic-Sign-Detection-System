import os
import yaml
import wandb
from ultralytics import YOLO
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "model": "yolov8n.pt", 
    "data_yaml": PROJECT_ROOT + "/dataset/vietnam_traffic_signs.yaml",
    "epochs": 100,
    "patience": 20,
    "imgsz": 640,
    "batch": 32,
    "device": "0", 
    "project": "yolo-viet-signs",
    "name": "yolov8n_viet_signs",
    "use_wandb": True,
    "wandb_key": "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",  
}

CLASS_NAMES = [
    'Đường người đi bộ cắt ngang',
    'Đường giao nhau (ngã ba bên phải)',
    'Cấm đi ngược chiều',
    'Phải đi vòng sang bên phải',
    'Giao nhau với đường đồng cấp',
    'Giao nhau với đường không ưu tiên',
    'Chỗ ngoặt nguy hiểm vòng bên trái',
    'Cấm rẽ trái',
    'Bến xe buýt',
    'Nơi giao nhau chạy theo vòng xuyến',
    'Cấm dừng và đỗ xe',
    'Chỗ quay xe',
    'Biển gộp làn đường theo phương tiện',
    'Đi chậm',
    'Cấm xe tải',
    'Đường bị thu hẹp về phía phải',
    'Giới hạn chiều cao',
    'Cấm quay đầu',
    'Cấm ô tô khách và ô tô tải',
    'Cấm rẽ phải và quay đầu',
    'Cấm ô tô',
    'Đường bị thu hẹp về phía trái',
    'Gồ giảm tốc phía trước',
    'Cấm xe hai và ba bánh',
    'Kiểm tra',
    'Chỉ dành cho xe máy*',
    'Chướng ngoại vật phía trước',
    'Trẻ em',
    'Xe tải và xe công*',
    'Cấm mô tô và xe máy',
    'Chỉ dành cho xe tải*',
    'Đường có camera giám sát',
    'Cấm rẽ phải',
    'Nhiều chỗ ngoặt nguy hiểm liên tiếp, chỗ đầu tiên sang phải',
    'Cấm xe sơ-mi rơ-moóc',
    'Cấm rẽ trái và phải',
    'Cấm đi thẳng và rẽ phải',
    'Đường giao nhau (ngã ba bên trái)',
    'Giới hạn tốc độ (50km/h)',
    'Giới hạn tốc độ (60km/h)',
    'Giới hạn tốc độ (80km/h)',
    'Giới hạn tốc độ (40km/h)',
    'Các xe chỉ được rẽ trái',
    'Chiều cao tĩnh không thực tế',
    'Nguy hiểm khác',
    'Đường một chiều',
    'Cấm đỗ xe',
    'Cấm ô tô quay đầu xe (được rẽ trái)',
    'Giao nhau với đường sắt có rào chắn',
    'Cấm rẽ trái và quay đầu xe',
    'Chỗ ngoặt nguy hiểm vòng bên phải',
    'Chú ý chướng ngại vật – vòng tránh sang bên phải'
]


def log_epoch_metrics(trainer):
    """Callback to log metrics to wandb"""
    epoch = trainer.epoch
    metrics = trainer.metrics
    loss = trainer.loss_items
    
    wandb.log({
        "epoch": epoch,
        "train/box_loss": float(loss[0]),
        "train/cls_loss": float(loss[1]),
        "train/dfl_loss": float(loss[2]),
        "metrics/precision": metrics.get("metrics/precision(B)", 0),
        "metrics/recall": metrics.get("metrics/recall(B)", 0),
        "metrics/mAP50": metrics.get("metrics/mAP50(B)", 0),
        "metrics/mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
        "lr/pg0": trainer.lf(epoch) * trainer.args.lr0,
    })


def train():
    """Train YOLO model"""
    
    # Initialize wandb
    use_wandb = CONFIG["use_wandb"]
    if use_wandb:
        try:
            if CONFIG["wandb_key"]:
                wandb.login(key=CONFIG["wandb_key"])
            wandb.init(project=CONFIG["project"], name=CONFIG["name"])
            print("WandB initialized!")
        except Exception as e:
            print(f"WandB error: {e}")
            use_wandb = False
    
    # Load model
    print(f"Loading model: {CONFIG['model']}")
    model = YOLO(CONFIG["model"])
    
    # Add wandb callback
    if use_wandb:
        model.add_callback("on_train_epoch_end", log_epoch_metrics)
    
    # Train
    print(f"\n{'='*50}")
    print(f"Model: {CONFIG['model']}")
    print(f"Data: {CONFIG['data_yaml']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch: {CONFIG['batch']}")
    print(f"Image size: {CONFIG['imgsz']}")
    print(f"Device: {CONFIG['device']}")
    print(f"{'='*50}\n")
    
    results = model.train(
        data=CONFIG["data_yaml"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        project=CONFIG["project"],
        name=CONFIG["name"],
        patience=CONFIG["patience"],
        device=CONFIG["device"],
    )
    
    if use_wandb:
        wandb.finish()
    
    print("\nTraining completed!")
    return results


if __name__ == "__main__":
    train()
