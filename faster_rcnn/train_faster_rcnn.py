import os
import sys
import json
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from tqdm import tqdm
import wandb
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset import TrafficSignDataset


NUM_CLASSES = 53


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes, pretrained=True):
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0
    num_batches = 0
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in pbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Skip batches with no valid targets
        valid_targets = [t for t in targets if t["boxes"].shape[0] > 0]
        if len(valid_targets) == 0:
            continue
            
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Reduce losses
        loss_value = losses.item()
        
        if not torch.isfinite(losses):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            continue
            
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss_value
        running_loss_classifier += loss_dict.get('loss_classifier', 0).item() if isinstance(loss_dict.get('loss_classifier', 0), torch.Tensor) else loss_dict.get('loss_classifier', 0)
        running_loss_box_reg += loss_dict.get('loss_box_reg', 0).item() if isinstance(loss_dict.get('loss_box_reg', 0), torch.Tensor) else loss_dict.get('loss_box_reg', 0)
        running_loss_objectness += loss_dict.get('loss_objectness', 0).item() if isinstance(loss_dict.get('loss_objectness', 0), torch.Tensor) else loss_dict.get('loss_objectness', 0)
        running_loss_rpn_box_reg += loss_dict.get('loss_rpn_box_reg', 0).item() if isinstance(loss_dict.get('loss_rpn_box_reg', 0), torch.Tensor) else loss_dict.get('loss_rpn_box_reg', 0)
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss_value:.4f}",
            'avg_loss': f"{running_loss / num_batches:.4f}"
        })
    
    avg_losses = {
        'total_loss': running_loss / max(num_batches, 1),
        'loss_classifier': running_loss_classifier / max(num_batches, 1),
        'loss_box_reg': running_loss_box_reg / max(num_batches, 1),
        'loss_objectness': running_loss_objectness / max(num_batches, 1),
        'loss_rpn_box_reg': running_loss_rpn_box_reg / max(num_batches, 1),
    }
    
    return avg_losses


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(data_loader, desc="Evaluating")
    
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        
        outputs = model(images)
        
        for output, target in zip(outputs, targets):
            all_predictions.append({
                'boxes': output['boxes'].cpu(),
                'scores': output['scores'].cpu(),
                'labels': output['labels'].cpu()
            })
            all_targets.append({
                'boxes': target['boxes'],
                'labels': target['labels']
            })
    
    # Simple evaluation metrics
    total_predictions = sum(len(p['boxes']) for p in all_predictions)
    total_gt = sum(len(t['boxes']) for t in all_targets)
    
    # Count predictions with high confidence
    high_conf_preds = sum(
        (p['scores'] > 0.5).sum().item() 
        for p in all_predictions
    )
    
    metrics = {
        'total_predictions': total_predictions,
        'total_ground_truth': total_gt,
        'high_conf_predictions': high_conf_preds,
    }
    
    return metrics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def main():
    config = {
        'batch_size': 4,
        'num_epochs': 50,
        'learning_rate': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_step_size': 10,
        'lr_gamma': 0.1,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',
        'pretrained': True,
    }
    
    train_images_dir = PROJECT_ROOT + "/dataset/images/train"
    train_labels_dir = PROJECT_ROOT + "/dataset/labels/train"
    val_images_dir = PROJECT_ROOT + "/dataset/images/val"
    val_labels_dir = PROJECT_ROOT + "/dataset/labels/val"
    
    print(f"Using device: {config['device']}")
    print(f"Number of classes: {NUM_CLASSES}")
    
    use_wandb = True
    try:
        wandb.init(
            project="faster-rcnn-viet-signs",
            name=f"frcnn_resnet50_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )
    except Exception as e:
        print(f"Could not initialize wandb: {e}")
        use_wandb = False
    
    train_dataset = TrafficSignDataset(train_images_dir, train_labels_dir, augment=True)
    val_dataset = TrafficSignDataset(val_images_dir, val_labels_dir, augment=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    model = get_model(NUM_CLASSES, pretrained=config['pretrained'])
    model.to(config['device'])
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*50}")
        
        train_losses = train_one_epoch(
            model, optimizer, train_loader, 
            config['device'], epoch
        )
        
        val_metrics = evaluate(model, val_loader, config['device'])
        
        lr_scheduler.step()
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_losses['total_loss']:.4f}")
        print(f"  - Classifier: {train_losses['loss_classifier']:.4f}")
        print(f"  - Box Reg: {train_losses['loss_box_reg']:.4f}")
        print(f"  - Objectness: {train_losses['loss_objectness']:.4f}")
        print(f"  - RPN Box Reg: {train_losses['loss_rpn_box_reg']:.4f}")
        print(f"  Val Predictions: {val_metrics['total_predictions']}")
        print(f"  Val High Conf Preds: {val_metrics['high_conf_predictions']}")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/total_loss': train_losses['total_loss'],
                'train/loss_classifier': train_losses['loss_classifier'],
                'train/loss_box_reg': train_losses['loss_box_reg'],
                'train/loss_objectness': train_losses['loss_objectness'],
                'train/loss_rpn_box_reg': train_losses['loss_rpn_box_reg'],
                'val/total_predictions': val_metrics['total_predictions'],
                'val/high_conf_predictions': val_metrics['high_conf_predictions'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if train_losses['total_loss'] < best_loss:
            best_loss = train_losses['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"  * Saved best model (loss: {best_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_losses['total_loss'],
            }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}.pth'))
            print(f"  * Saved checkpoint")
    
    # Save final model
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_losses['total_loss'],
    }, os.path.join(config['save_dir'], 'final_model.pth'))
    
    if use_wandb:
        wandb.finish()
    
    print("\nTraining completed!")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
