import sys
sys.path.append('../')
import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from configs.defaults import get_cfg_defaults
from torch.optim.lr_scheduler import MultiStepLR
from data.slp_dataset import SLPDataset
from model.dgcnn import DGCNN
from model.da_utils import Augmenter, get_pose_pred, get_current_consistency_weight, compute_angle_loss,\
    compute_symmetry_loss, compute_range_loss, update_ema_variables


def train_uda(cfg, output_directory):
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
    train_set_source.data_list = train_set_source.data_list[:24]

    # model
    student = DGCNN(cfg).to(cfg.MODEL.DEVICE)
    teacher = DGCNN(cfg).to(cfg.MODEL.DEVICE)
    for param in teacher.parameters():
        param.requires_grad = False

    # optimizer and scheduler
    use_amp = cfg.SOLVER.USE_AMP # whether to use mixed precision
    optimizer = Adam(student.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    lr_scheduler = MultiStepLR(optimizer, cfg.SOLVER.LR_MILESTONES, cfg.SOLVER.LR_GAMMA)

    # parameters for domain adaptation
    # anatomical constraints
    bone_upper = torch.tensor(cfg.DA.ANATOMY_BONE_LENGTH_UPPER, device=cfg.MODEL.DEVICE).float().unsqueeze(0)
    bone_lower = torch.tensor(cfg.DA.ANATOMY_BONE_LENGTH_LOWER, device=cfg.MODEL.DEVICE).float().unsqueeze(0)
    angle_upper = torch.tensor(cfg.DA.ANATOMY_ANGLE_UPPER, device=cfg.MODEL.DEVICE).float().unsqueeze(0)
    angle_lower = torch.tensor(cfg.DA.ANATOMY_ANGLE_LOWER, device=cfg.MODEL.DEVICE).float().unsqueeze(0)
    symmetry_factor = cfg.DA.ANATOMY_SYMMETRY
    bone_factor = cfg.DA.ANATOMY_RANGE
    angle_factor = cfg.DA.ANATOMY_ANGLE
    # mean teacher
    augmenter = Augmenter(cfg)
    cons_loss_fac = cfg.DA.MEAN_TEACHER.CONS_LOSS_FAC
    rampup_len = cfg.DA.MEAN_TEACHER.RAMPUP_LENGTH
    teacher_alpha = cfg.DA.MEAN_TEACHER.ALPHA


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
        student.train()
        teacher.train()
        loss_values_s = []
        loss_values_da = []
        start_time = time.time()
        train_loader_target_iter = iter(train_loader_target)
        for it, data_s in enumerate(train_loader_source, 1):
            pcd_s, joints_s, idx_s = data_s
            try:
                data_t = next(train_loader_target_iter)
            except StopIteration:
                train_loader_target_iter = iter(train_loader_target)
                data_t = next(train_loader_target_iter)
            pcd_t, _, idx_t = data_t

            inputs_s = pcd_s.to(cfg.MODEL.DEVICE)
            targets_s = joints_s.to(cfg.MODEL.DEVICE)
            inputs_t = pcd_t.to(cfg.MODEL.DEVICE)

            # augmentations
            x_t_augm1, rot_mat1, transl1 = augmenter(inputs_t)
            x_t_augm2, rot_mat2, transl2 = augmenter(inputs_t)
            B_t = inputs_t.size(0)
            B_s = inputs_s.size(0)
            inputs_student = torch.cat((inputs_s, x_t_augm1), dim=0)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_stud, _ = student(inputs_student)
                pred_s_stud = pred_stud[:B_s]
                pred_t_stud = pred_stud[B_s:]
                pred_teach, _ = teacher(x_t_augm2)

                loss_s = criterion(inputs_s, pred_s_stud, targets_s)

                pose_pred = get_pose_pred(x_t_augm1, pred_t_stud)
                pose_pred = torch.bmm(rot_mat1, pose_pred.transpose(1, 2) - transl1).transpose(1, 2)
                teacher_pose_pred = get_pose_pred(x_t_augm2, pred_teach)
                teacher_pose_pred = torch.bmm(rot_mat2, teacher_pose_pred.transpose(1, 2) - transl2).transpose(1, 2)

                bone_loss_teach = compute_range_loss(teacher_pose_pred, range_min=bone_lower,
                                                     range_max=bone_upper, reduce='none').mean(dim=1)
                angle_loss_teach = compute_angle_loss(teacher_pose_pred, range_min=angle_lower,
                                                      range_max=angle_upper, reduce='none').mean(dim=1)
                sym_loss_teach = compute_symmetry_loss(teacher_pose_pred, reduce='none').mean(dim=1)
                bone_loss_stud = compute_range_loss(pose_pred, range_min=bone_lower,
                                                    range_max=bone_upper, reduce='none').mean(dim=1)
                angle_loss_stud = compute_angle_loss(pose_pred, range_min=angle_lower,
                                                     range_max=angle_upper, reduce='none').mean(dim=1)
                sym_loss_stud = compute_symmetry_loss(pose_pred, reduce='none').mean(dim=1)


                loss_coef = ((bone_loss_teach < bone_loss_stud).float() + (angle_loss_teach < angle_loss_stud).float() + (sym_loss_teach < sym_loss_stud).float()) > 1.5
                coeff_fac = loss_coef.sum() / loss_coef.numel()
                ema_loss_fac_adj = torch.nan_to_num(1 / coeff_fac, posinf=0.)
                ema_loss = (torch.abs(pose_pred - teacher_pose_pred).mean(dim=(1, 2)) * loss_coef).mean()
                ema_loss = ema_loss_fac_adj * ema_loss

                ema_loss += bone_factor * bone_loss_stud.mean()
                ema_loss += symmetry_factor * sym_loss_stud.mean()
                ema_loss += angle_factor * angle_loss_stud.mean()

                loss = loss_s + get_current_consistency_weight(cons_loss_fac, e + 1, rampup_len) * ema_loss


            loss_values_s.append(loss_s.item())
            loss_values_da.append(ema_loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            update_ema_variables(student, teacher, teacher_alpha)

        train_loss_s = np.mean(loss_values_s)
        train_loss_da = np.mean(loss_values_da)
        lr_scheduler.step()

        # Validation
        val_scores = []
        for model in [student, teacher]:
            model.eval()
            for loader, domain in zip([val_loader_source, val_loader_target], ['source', 'target']):
                mpjpe = 0
                for it, data in enumerate(loader, 1):
                    pcd, joints, idx = data
                    inputs = pcd.to(cfg.MODEL.DEVICE)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        with torch.no_grad():
                            pred, _ = model(inputs)
                    p = F.softmax(pred, 2).unsqueeze(2)
                    pred_pose = (p * inputs.unsqueeze(1)).sum(3)
                    joints = joints.to(cfg.MODEL.DEVICE)
                    error = (joints - pred_pose).norm(dim=2).sum().item()
                    mpjpe += error

                num_joints = joints.shape[1]
                mpjpe = mpjpe / (len(loader.dataset) * num_joints)
                val_scores.append(mpjpe)

        end_time = time.time()
        print('epoch', e, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'train_loss_s', '%0.6f' % train_loss_s,
              'train_loss_da', '%0.6f' % train_loss_da, 'MPJPE_s_stud', '%0.6f' % val_scores[0], 'MPJPE_s_teach', val_scores[2],
              'MPJPE_t_stud', '%0.6f' % val_scores[1], 'MPJPE_t_teach', val_scores[3],)

        validation_log[e, :] = [train_loss_s, train_loss_da, val_scores[0], val_scores[2], val_scores[1], val_scores[3]]
        np.save(os.path.join(output_directory, "validation_history"), validation_log)
        torch.save(teacher.state_dict(), os.path.join(output_directory, 'model.pth'))


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

    train_uda(cfg, output_directory)