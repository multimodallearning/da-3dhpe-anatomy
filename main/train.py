import sys
sys.path.append('../')
sys.path.append('../models')
import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from configs.defaults import get_cfg_defaults
from torch.optim.lr_scheduler import MultiStepLR
from data.slp_dataset import SLPDataset
from model.dgcnn import DGCNN


def train(cfg, output_directory):
    if cfg.DA.METHOD == 'oracle':
        train_set_source = SLPDataset(cfg, domain='target', phase='train', split='train')
    else:
        train_set_source = SLPDataset(cfg, domain='source', phase='train', split='train')
    train_loader_source = DataLoader(train_set_source, batch_size=cfg.SOLVER.BATCH_SIZE_TRAIN,
                                     num_workers=cfg.INPUT.NUM_WORKERS, shuffle=True, drop_last=True)
    train_set_target = SLPDataset(cfg, domain='target', phase='train', split='train')
    train_loader_target = DataLoader(train_set_target, batch_size=cfg.SOLVER.BATCH_SIZE_TRAIN,
                                     num_workers=cfg.INPUT.NUM_WORKERS, shuffle=True, drop_last=True)
    val_set_source = SLPDataset(cfg, domain='source', phase='val', split='val')
    val_loader_source = DataLoader(val_set_source, batch_size=cfg.SOLVER.BATCH_SIZE_TEST,
                                   num_workers=cfg.INPUT.NUM_WORKERS, shuffle=False, drop_last=False)
    val_set_target = SLPDataset(cfg, domain='target', phase='val', split='val')
    val_loader_target = DataLoader(val_set_target, batch_size=cfg.SOLVER.BATCH_SIZE_TEST,
                                   num_workers=cfg.INPUT.NUM_WORKERS, shuffle=False, drop_last=False)

    # model
    model = DGCNN(cfg).to(cfg.MODEL.DEVICE)
    if cfg.MODEL.WEIGHT != '':
        model.load_state_dict(torch.load(cfg.MODEL.WEIGHT), strict=False)

    # optimizer and scheduler
    use_amp = cfg.SOLVER.USE_AMP # whether to use mixed precision
    optimizer = Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    lr_scheduler = MultiStepLR(optimizer, cfg.SOLVER.LR_MILESTONES, cfg.SOLVER.LR_GAMMA)

    # loss criterion
    def criterion(input, pred, pose):
        p = F.softmax(pred, 2).unsqueeze(2)
        pose_pred = (p * input.unsqueeze(1)).sum(3)
        loss = nn.L1Loss()
        return loss(pose_pred, pose)

    # logging: save training and validation accuracy after each epoch to numpy array
    validation_log = np.zeros([cfg.SOLVER.NUM_EPOCHS, 6])

    # Training
    for e in range(cfg.SOLVER.NUM_EPOCHS):
        model.train()
        loss_values_s = []
        loss_values_da = []
        start_time = time.time()
        train_loader_target_iter = iter(train_loader_target)
        for it, data_s in enumerate(train_loader_source, 1):
            pcd_s, joints_s, idx_s = data_s
            data_t = next(train_loader_target_iter)
            pcd_t, _, idx_t = data_t

            inputs_s = pcd_s.to(cfg.MODEL.DEVICE)
            targets_s = joints_s.to(cfg.MODEL.DEVICE)
            inputs_t = pcd_t.to(cfg.MODEL.DEVICE)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_s, loss_da = model(inputs_s, inputs_t)
                loss_s = criterion(inputs_s, pred_s, targets_s)

            loss = loss_s + loss_da
            loss_values_s.append(loss_s.item())
            loss_values_da.append(loss_da.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss_s = np.mean(loss_values_s)
        train_loss_da = np.mean(loss_values_da)
        lr_scheduler.step()

        # Validation
        model.eval()
        for loader, domain in zip([val_loader_source, val_loader_target], ['source', 'target']):
            loss_values = []
            mpjpe = 0
            for it, data in enumerate(loader, 1):
                pcd, joints, idx = data
                inputs = pcd.to(cfg.MODEL.DEVICE)
                targets = joints.to(cfg.MODEL.DEVICE)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    with torch.no_grad():
                        pred, _ = model(inputs)
                    loss_values.append(criterion(inputs, pred, targets).item())
                    p = F.softmax(pred, 2).unsqueeze(2)
                    pred_pose = (p * inputs.unsqueeze(1)).sum(3)

                joints = joints.to(cfg.MODEL.DEVICE)
                error = (joints - pred_pose).norm(dim=2).sum().item()
                mpjpe += error

            if domain == 'source':
                val_loss_s = np.mean(loss_values)
                mpjpe_s = mpjpe / (len(val_set_source) * cfg.OUTPUT.NUM_JOINTS)
            elif domain == 'target':
                val_loss_t = np.mean(loss_values)
                mpjpe_t = mpjpe / (len(val_set_target) * cfg.OUTPUT.NUM_JOINTS)

        end_time = time.time()
        print('epoch', e, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'train_loss_s', '%0.6f' % train_loss_s,
              'val_loss_s', '%0.6f' % val_loss_s, 'MPJPE_s', mpjpe_s, 'train_loss_da', '%0.6f' % train_loss_da)
        validation_log[e, :] = [train_loss_s, val_loss_s, mpjpe_s, train_loss_da, val_loss_t, mpjpe_t]

        np.save(os.path.join(output_directory, "validation_history"), validation_log)
        torch.save(model.state_dict(), os.path.join(output_directory, 'model.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default="0",
        metavar="FILE",
        help="gpu to train on",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_directory = os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    train(cfg, output_directory)
