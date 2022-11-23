import sys
sys.path.append('../')
import argparse
import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from configs.defaults import get_cfg_defaults
from torch.optim.lr_scheduler import MultiStepLR
from data.slp_dataset import SLPDataset
from model.dgcnn import DGCNN

from model.da_utils import get_pose_pred, Augmenter, compute_symmetry_loss, compute_range_loss, compute_angle_loss, update_ema_variables


def train_sfda(cfg, output_directory):
    # datasets
    train_set_target = SLPDataset(cfg, domain='target', phase='train', split='train')
    val_set_target = SLPDataset(cfg, domain='target', phase='val', split='val')
    train_loader_target = DataLoader(train_set_target, batch_size=cfg.SOLVER.BATCH_SIZE_TRAIN,
                                     num_workers=cfg.INPUT.NUM_WORKERS, shuffle=True, drop_last=True)
    val_loader_target = DataLoader(val_set_target, batch_size=cfg.SOLVER.BATCH_SIZE_TEST,
                                   num_workers=cfg.INPUT.NUM_WORKERS, shuffle=False, drop_last=False)

    # model setup
    student = DGCNN(cfg).to(cfg.MODEL.DEVICE)
    teacher = DGCNN(cfg).to(cfg.MODEL.DEVICE)
    if cfg.MODEL.WEIGHT != '':
        student.load_state_dict(torch.load(cfg.MODEL.WEIGHT), strict=False)
        teacher.load_state_dict(torch.load(cfg.MODEL.WEIGHT), strict=False)
    else:
        raise ValueError('Training in SFDA setting requires loading weights of a pre-trained source model.')

    # freeze parameters of the network heads of the student
    for layer in [student.conv7, student.conv8, student.conv9]:
        for name, param in layer.named_parameters():
            param.requires_grad = False
    # freeze all parameters of the teacher model
    for param in teacher.parameters():
        param.requires_grad = False

    # hyper-parameters
    # mean teacher
    augmenter = Augmenter(cfg)
    teacher_alpha = cfg.DA.MEAN_TEACHER.ALPHA
    cons_loss_fac = cfg.DA.MEAN_TEACHER.CONS_LOSS_FAC

    # anatomical constraints
    symmetry_factor = cfg.DA.ANATOMY_SYMMETRY
    bone_factor = cfg.DA.ANATOMY_RANGE
    angle_factor = cfg.DA.ANATOMY_ANGLE
    bone_upper = torch.tensor(cfg.DA.ANATOMY_BONE_LENGTH_UPPER, device=cfg.MODEL.DEVICE).float().unsqueeze(0)
    bone_lower = torch.tensor(cfg.DA.ANATOMY_BONE_LENGTH_LOWER, device=cfg.MODEL.DEVICE).float().unsqueeze(0)
    angle_upper = torch.tensor(cfg.DA.ANATOMY_ANGLE_UPPER, device=cfg.MODEL.DEVICE).float().unsqueeze(0)
    angle_lower = torch.tensor(cfg.DA.ANATOMY_ANGLE_LOWER, device=cfg.MODEL.DEVICE).float().unsqueeze(0)

    # optimizer and scheduler
    use_amp = cfg.SOLVER.USE_AMP # whether to use mixed precision
    optimizer = Adam(student.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    lr_scheduler = MultiStepLR(optimizer, cfg.SOLVER.LR_MILESTONES, cfg.SOLVER.LR_GAMMA)

    # logging: save training and validation accuracy after each epoch to numpy array
    validation_log = np.zeros([cfg.SOLVER.NUM_EPOCHS, 3])

    # Training routine
    for e in range(cfg.SOLVER.NUM_EPOCHS):
        student.train()
        teacher.train()

        loss_values = []
        start_time = time.time()
        for it, data in enumerate(train_loader_target, 1):
            pcd, joints, idx = data
            inputs = pcd.to(cfg.MODEL.DEVICE)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = 0
                # student predictions
                x_augm, rot_mat, transl = augmenter(inputs)
                pred, _ = student(x_augm)
                pose_pred = get_pose_pred(x_augm, pred)
                pose_pred = torch.bmm(rot_mat, pose_pred.transpose(1, 2) - transl).transpose(1, 2)

                # teacher predictions
                x_augm_teach, rot_mat_teach, transl_teach = augmenter(inputs)
                with torch.no_grad():
                    pred_teach, _ = teacher(x_augm_teach)
                pose_pred_teach = get_pose_pred(x_augm_teach, pred_teach)
                pose_pred_teach = torch.bmm(rot_mat_teach, pose_pred_teach.transpose(1, 2) - transl_teach).transpose(1, 2)

                # filter pseudo labels by teacher according to anatomical losses
                bone_loss_teach = compute_range_loss(pose_pred_teach, range_min=bone_lower,
                                               range_max=bone_upper, reduce='none').mean(dim=1)
                angle_loss_teach = compute_angle_loss(pose_pred_teach, range_min=angle_lower,
                                                range_max=angle_upper, reduce='none').mean(dim=1)
                sym_loss_teach = compute_symmetry_loss(pose_pred_teach, reduce='none').mean(dim=1)
                bone_loss_stud = compute_range_loss(pose_pred, range_min=bone_lower,
                                                 range_max=bone_upper, reduce='none').mean(dim=1)
                angle_loss_stud = compute_angle_loss(pose_pred, range_min=angle_lower,
                                                  range_max=angle_upper, reduce='none').mean(dim=1)
                sym_loss_stud = compute_symmetry_loss(pose_pred, reduce='none').mean(dim=1)

                # reweight the selected samples such that the magnitude of the teacher loss remains roughly constant independent of the number of selected pseudo labels
                loss_coef = ((bone_loss_teach < bone_loss_stud).float() + (angle_loss_teach < angle_loss_stud).float() + (sym_loss_teach < sym_loss_stud).float()) > 1.5
                coeff_fac = loss_coef.sum() / loss_coef.numel()
                ema_loss_fac_adj = torch.nan_to_num(1 / coeff_fac, posinf=0.)

                ema_loss = (torch.abs(pose_pred - pose_pred_teach).mean(dim=(1, 2)) * loss_coef).mean()
                loss += cons_loss_fac * ema_loss_fac_adj * ema_loss

                # add anatomical loss terms
                loss += bone_factor * bone_loss_stud.mean()
                loss += symmetry_factor * sym_loss_stud.mean()
                loss += angle_factor * angle_loss_stud.mean()

            loss_values.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_ema_variables(student, teacher, teacher_alpha)

        train_loss = np.mean(loss_values)
        lr_scheduler.step()

        # Validation
        val_mpjpes = []
        for model in [student, teacher]:
            model.eval()
            mpjpe = 0
            for it, data in enumerate(val_loader_target, 1):
                pcd, joints, idx = data
                inputs = pcd.to(cfg.MODEL.DEVICE)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    with torch.no_grad():
                        pred, _ = model(inputs)
                pred_pose = get_pose_pred(inputs, pred)
                joints = joints.to(cfg.MODEL.DEVICE)
                error = (joints - pred_pose).norm(dim=2).sum().item()
                mpjpe += error

            num_joints = joints.shape[1]
            mpjpe /= (len(val_set_target) * num_joints)
            val_mpjpes.append(mpjpe)
        train_loss = 0.
        end_time = time.time()
        print('epoch', e, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'loss', '%0.6f' % train_loss,
              'MPJPE stud', val_mpjpes[0], 'MPJPE teach', val_mpjpes[1],)

        validation_log[e, :] = [train_loss, val_mpjpes[0], val_mpjpes[1]]
        np.save(os.path.join(output_directory, "validation_history.npy"), validation_log)
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

    train_sfda(cfg, output_directory)