import sys, os
import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

from logger_epoch import WandbLogger

import wandb


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Name for the wandb run (optional)')
    
    # loss 가중치 적용을 위한 인자 추가
    parser.add_argument('--cls_loss_weight', type=float, default=1.0)
    parser.add_argument('--angle_loss_weight', type=float, default=1.0)
    parser.add_argument('--iou_loss_weight', type=float, default=1.0)
    
    # 데이터 증강 확률을 위한 인자
    parser.add_argument('--augmentation_prob', type=float, default=0.0)
    

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    # wandb sweep이 있다면 해당 설정으로 덮어쓰기
    if wandb.config:
        args.learning_rate = wandb.config.learning_rate
        args.batch_size = wandb.config.batch_size
        args.weight_decay = wandb.config.weight_decay
        args.optimizer = wandb.config.optimizer
        args.image_size = wandb.config.image_size
        args.input_size = wandb.config.input_size
        args.cls_loss_weight = wandb.config.cls_loss_weight
        args.angle_loss_weight = wandb.config.angle_loss_weight
        args.iou_loss_weight = wandb.config.iou_loss_weight
        args.augmentation_prob = wandb.config.augmentation_prob
        args.max_epoch = wandb.config.max_epoch
        
    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, weight_decay, optimizer, max_epoch, save_interval, wandb_run_name):
    
    # Initialize wandb logger
    logger = WandbLogger(name=wandb_run_name)
    
    # Initialize wandb with config
    config = {
        "learning_rate": learning_rate,
        "max_epoch": max_epoch,
        "batch_size": batch_size,
        "image_size": image_size,
        "input_size": input_size,
        "optimizer": "AdamW",  # 실제 사용하는 옵티마이저로 수정
        "scheduler": "CosineAnnealingLR",
        "device": device
    }
    logger.initialize(config)
    
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # device 설정 중복 제거
    device = torch.device(device)
    model = EAST()
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

    model.train()
    
    for epoch in range(max_epoch):
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_angle_loss = 0
        epoch_iou_loss = 0
        epoch_start = time.time()
        
        with tqdm(total=num_batches) as pbar:
            for batch_idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
                # 현재 에폭과 배치 진행상황을 보여주는 description
                pbar.set_description(
                    f'[Epoch {epoch+1}/{max_epoch}][{batch_idx+1}/{num_batches}]'
                )

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 배치 크기를 고려한 loss 누적
                batch_size = img.size(0)  # 실제 배치 크기 (마지막 배치는 더 작을 수 있음)
                loss_val = loss.item() * batch_size
                epoch_loss += loss_val
                epoch_cls_loss += extra_info['cls_loss'] * batch_size
                epoch_angle_loss += extra_info['angle_loss'] * batch_size
                epoch_iou_loss += extra_info['iou_loss'] * batch_size

                pbar.update(1)
                val_dict = {
                    'Loss': f"{loss_val/batch_size:.4f}",
                    'Cls': f"{extra_info['cls_loss']:.4f}", 
                    'Angle': f"{extra_info['angle_loss']:.4f}",
                    'IoU': f"{extra_info['iou_loss']:.4f}",
                    'LR': f"{scheduler.get_last_lr()[0]:.6f}"
                }
                pbar.set_postfix(val_dict)

        scheduler.step()
        
        # 에폭 완료 시간 계산
        epoch_time = timedelta(seconds=time.time() - epoch_start)
        
        # 전체 데이터셋 크기로 나누어 평균 계산
        dataset_size = len(dataset)
        mean_epoch_loss = epoch_loss / dataset_size
        mean_cls_loss = epoch_cls_loss / dataset_size
        mean_angle_loss = epoch_angle_loss / dataset_size
        mean_iou_loss = epoch_iou_loss / dataset_size

        # 에폭 단위로 모든 메트릭 로깅
        logger.log_epoch_metrics(
            {
                "epoch": epoch,
                "loss": epoch_loss / num_batches,
                "cls_loss": epoch_cls_loss / num_batches,
                "angle_loss": epoch_angle_loss / num_batches,
                "iou_loss": epoch_iou_loss / num_batches,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_time": timedelta(seconds=time.time() - epoch_start).total_seconds()
            }
        )

        print(f'Epoch {epoch+1}/{max_epoch} | Mean loss: {mean_epoch_loss:.4f} | Time: {epoch_time}')

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': mean_epoch_loss,
                }, 
                ckpt_fpath
            )
            
            # Log model checkpoint
            logger.log_model(ckpt_fpath, f'model-epoch-{epoch+1}')

    logger.finish()
    

def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
#python train_wandb.py  # 기본 실행
#python train_wandb.py --wandb_run_name "experiment-1"  # 실험 이름 지정하고 싶을 때
