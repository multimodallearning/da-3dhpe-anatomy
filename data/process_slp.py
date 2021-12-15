# Here, we generate point clouds from the depth maps in the SLP dataset according to Clever et al. and following the
# SLP-3Dfits repo (https://github.com/pgrady3/SLP-3Dfits). Specifically, the point cloud is warped to the pressure maps.
# This is necessary to ensure that point clouds are in accordance with the ground truth SMPL fits

import sys
sys.path.append('../')
sys.path.append('../../SLP-Dataset-and-Code')
import os
import pickle
import numpy as np
import torch
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from data.SLP_RD import SLP_RD
import utils.utils as ut    # SLP utils


def apply_homography(points, h, yx=True):
    # Apply 3x3 homography matrix to points
    # Note that the homography matrix is parameterized as XY,
    # but all image coordinates are YX

    if yx:
        points = np.flip(points, 1)

    points_h = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
    tform_h = np.matmul(h, points_h.T).T
    tform_h = tform_h / tform_h[:, 2][:, np.newaxis]

    points = tform_h[:, :2]

    if yx:
        points = np.flip(points, 1)

    return points


def get_modified_depth_to_pressure_homography(slp_dataset, idx):
    """
    Magic function to get the homography matrix to warp points in depth-cloud space into pressure-mat space.
    However, this modifies the scaling of the homography matrix to keep the same scale, so that the points can be
    projected into 3D metric space.
    :param slp_dataset: Dataset object to extract homography from
    :param idx: SLP dataset sample index
    :return:
    """
    WARPING_MAGIC_SCALE_FACTOR = (192. / 345.)  # Scale matrix to align to PM. 192 is height of pressure mat, 345 is height of bed in depth pixels

    depth_Tr = slp_dataset.get_PTr_A2B(idx=idx, modA='depthRaw', modB='PM')     # Get SLP homography matrix
    depth_Tr /= depth_Tr[2, 2]  # Make more readable matrix

    depth_Tr[0:2, 0:3] = depth_Tr[0:2, 0:3] / WARPING_MAGIC_SCALE_FACTOR
    return depth_Tr


def project_depth_with_warping(slp_dataset, depth_arr, idx):
    """
    Project a 2D depth image into 3D space. Additionally, this warps the 2D points of the input image
    using a homography matrix. Importantly, there is no interpolation in the warping step.
    :param slp_dataset: Dataset object to extract image from
    :param depth_arr: The input depth image to use
    :param idx: SLP dataset sample index
    :return: A [N, 3] numpy array of the 3D pointcloud
    """
    # The other methods of using cv2.warpPerspective apply interpolation to the image, which is bad. This doesn't
    # Input image is YX
    depth_homography = get_modified_depth_to_pressure_homography(slp_dataset, idx)
    orig_x, orig_y = np.meshgrid(np.arange(0, depth_arr.shape[1]), np.arange(0, depth_arr.shape[0]))

    image_space_coordinates = np.stack((orig_x.flatten(), orig_y.flatten()), 0).T
    warped_image_space_coordinates = apply_homography(image_space_coordinates, depth_homography, yx=False)

    cd_modified = np.matmul(depth_homography, np.array([slp_dataset.c_d[0], slp_dataset.c_d[1], 1.0]).T)    # Multiply the center of the depth image by homography
    cd_modified = cd_modified/cd_modified[2]    # Re-normalize

    projection_input = np.concatenate((warped_image_space_coordinates, depth_arr.flatten()[..., np.newaxis]), 1)
    ptc = ut.pixel2cam(projection_input, slp_dataset.f_d, cd_modified[0:2]) / 1000.0
    return ptc



def get_depth_henry(idx, sample, dataset):
    # Get the depth image, but warped to PM
    raw_depth = dataset.get_array_A2B(idx=idx, modA='depthRaw', modB='depthRaw')

    pointcloud = project_depth_with_warping(dataset, raw_depth, idx)

    valid_z = np.logical_and(pointcloud[:, 2] > 1.55, pointcloud[:, 2] < 2.15)  # Cut out any outliers above the bed
    valid_x = np.logical_and(pointcloud[:, 0] > -0.3, pointcloud[:, 0] < 0.8)  # Cut X
    valid_y = np.logical_and(pointcloud[:, 1] > -1.1, pointcloud[:, 1] < 1.0)  # Cut Y
    valid_all = np.logical_and.reduce((valid_x, valid_y, valid_z))
    pointcloud = pointcloud[valid_all, :]

    return pointcloud


def prepare_pcds():
    target_directory = '../datasets/SLP_processed/3d_data_{}_{}'
    target_file = 'danaLab_{:05d}_{}_{:03d}_bed_pcd.npy'

    class PseudoOpts:
        SLP_fd = '../datasets/SLP/danaLab'
        sz_pch = [256, 256]
        fc_depth = 50
        cov_li = ['uncover', 'cover1', 'cover2']

    SLP_dataset = SLP_RD(PseudoOpts, phase='all')
    all_samples = SLP_dataset.pthDesc_li
    for idx, sample in enumerate(all_samples):
        print(sample)
        pcd = get_depth_henry(idx, sample, SLP_dataset)

        subj_no, cover_cond, pose_no = sample
        if pose_no <= 15:
            position = 'supine'
        elif pose_no <= 30:
            position = 'left'
        else:
            position = 'right'

        dir = target_directory.format(position, cover_cond)
        file = target_file.format(subj_no, cover_cond, pose_no)

        if not os.path.isdir(dir):
            os.makedirs(dir)
        path = os.path.join(dir, file)
        np.save(path, pcd)


def prepare_gt_joints(smpl_path):
    target_directory_joints = '../datasets/SLP_processed/gt_joints_3d'
    if not os.path.isdir(target_directory_joints):
        os.makedirs(target_directory_joints)
    target_file_joints = 'subject_{:03d}_pose_{:02d}_gt_joints.npy'
    target_path_joints = os.path.join(target_directory_joints, target_file_joints)

    source_template = '../../SLP-3Dfits/fits/p{:03d}/s{:02d}.pkl'

    class PseudoOpts:
        SLP_fd = '../datasets/SLP/danaLab'
        sz_pch = [256, 256]
        fc_depth = 50
        cov_li = ['uncover']

    SLP_dataset = SLP_RD(PseudoOpts, phase='all')
    all_samples = SLP_dataset.pthDesc_li
    for idx, sample in enumerate(all_samples):
        print(sample)
        subj_no, _, pose_no = sample
        pkl_data = pickle.load(open(source_template.format(subj_no, pose_no), 'rb'))
        smpl_layer = SMPL_Layer(
            center_idx=0,
            gender=pkl_data['gender'],
            model_root=smpl_path)
        pose_params = torch.Tensor(pkl_data['body_pose']).unsqueeze(0)
        betas = torch.Tensor(pkl_data['betas']).unsqueeze(0)
        transl = torch.Tensor(pkl_data['transl']).unsqueeze(0)
        global_orient = torch.Tensor(pkl_data['global_orient']).unsqueeze(0)
        pose_params = torch.cat((global_orient, pose_params), dim=1)
        _, joints = smpl_layer(pose_params, th_betas=betas, th_trans=transl)
        joints = joints.squeeze(0).numpy()
        joints_path = target_path_joints.format(subj_no, pose_no)
        np.save(joints_path, joints)


if __name__ == "__main__":
    smpl_path = '../../smpl_models'
    prepare_pcds()
    prepare_gt_joints(smpl_path)
