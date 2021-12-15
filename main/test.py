import sys
sys.path.append('../')
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.dgcnn import DGCNN
from data.slp_dataset import SLPDataset
from configs.defaults import get_cfg_defaults


def test(cfg, model_path, data_split, cover_condition):
    # data
    if cover_condition == 'cover12':
        cover_conds = ['cover1', 'cover2']
    else:
        cover_conds = [cover_condition]
    val_set = SLPDataset(cfg, phase='val', split=data_split, domain='target', cover_conds=cover_conds)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=cfg.INPUT.NUM_WORKERS,
                            shuffle=False, drop_last=False)

    # model
    model = DGCNN(cfg).to(cfg.MODEL.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    use_amp = cfg.SOLVER.USE_AMP

    # evaluation
    mpjpe = 0
    mpjpe_feet = 0
    mpjpe_knees = 0
    mpjpe_hips = 0
    mpjpe_core = 0
    mpjpe_head = 0
    mpjpe_shoulders = 0
    mpjpe_elbows = 0
    mpjpe_hands = 0
    for it, data in enumerate(val_loader, 1):
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
        mpjpe_feet += (joints[:, [7, 8, 10, 11]] - pred_pose[:, [7, 8, 10, 11]]).norm(dim=2).sum().item()
        mpjpe_knees += (joints[:, [4, 5]] - pred_pose[:, [4, 5]]).norm(dim=2).sum().item()
        mpjpe_hips += (joints[:, [1, 2]] - pred_pose[:, [1, 2]]).norm(dim=2).sum().item()
        mpjpe_core += (joints[:, [0, 3, 6, 9]] - pred_pose[:, [0, 3, 6, 9]]).norm(dim=2).sum().item()
        mpjpe_head += (joints[:, [12, 15]] - pred_pose[:, [12, 15]]).norm(dim=2).sum().item()
        mpjpe_shoulders += (joints[:, [13, 14, 16, 17]] - pred_pose[:, [13, 14, 16, 17]]).norm(dim=2).sum().item()
        mpjpe_elbows += (joints[:, [18, 19]] - pred_pose[:, [18, 19]]).norm(dim=2).sum().item()
        mpjpe_hands += (joints[:, [20, 21, 22, 23]] - pred_pose[:, [20, 21, 22, 23]]).norm(dim=2).sum().item()

    mpjpe /= (len(val_set) * cfg.OUTPUT.NUM_JOINTS)
    mpjpe_feet /= (len(val_set) * 4)
    mpjpe_knees /= (len(val_set) * 2)
    mpjpe_hips /= (len(val_set) * 2)
    mpjpe_core /= (len(val_set) * 4)
    mpjpe_head /= (len(val_set) * 2)
    mpjpe_shoulders /= (len(val_set) * 4)
    mpjpe_elbows /= (len(val_set) * 2)
    mpjpe_hands /= (len(val_set) * 4)

    print('MPJPE: ', mpjpe)
    print('MPJPE feet: ', mpjpe_feet)
    print('MPJPE knees: ', mpjpe_knees)
    print('MPJPE hips: ', mpjpe_hips)
    print('MPJPE core: ', mpjpe_core)
    print('MPJPE head: ', mpjpe_head)
    print('MPJPE shouldes: ', mpjpe_shoulders)
    print('MPJPE elbows: ', mpjpe_elbows)
    print('MPJPE hands: ', mpjpe_hands)


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
    parser.add_argument(
        "--model-path",
        default="",
        metavar="FILE",
        help="path to pre-trained model",
        type=str,
    )
    parser.add_argument(
        "--data-split",
        default="test",
        metavar="FILE",
        help="The data split to evaluate on. Should be in {val, test}.",
        type=str,
    )
    parser.add_argument(
        "--cover-condition",
        default="cover12",
        metavar="FILE",
        help="The cover condition to evaluate on. Should be in {uncover, cover1, cover2, cover12}.",
        type=str,
    )
    parser.add_argument(
        "--position",
        default="all",
        metavar="FILE",
        help="The position to evaluate on. Should be in {all, lateral, supine, left, right}",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    opts = ["SLP_DATASET.POSITION", args.position]
    cfg.merge_from_list(opts)
    cfg.freeze()

    if args.model_path == "":
        working_directory = os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)
        model_path = os.path.join(working_directory, 'model.pth')
    else:
        model_path = args.model_path
    if not os.path.isfile(model_path):
        raise ValueError('There is no pre-trained model at the specified path. Set the model path correctly or'
                         ' run training first.')

    test(cfg, model_path, args.data_split, args.cover_condition)