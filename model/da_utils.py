import torch
import torch.nn as nn
import torch.nn.functional as F


def get_pose_pred(x_in, x_out):
    p1 = F.softmax(x_out, 2).unsqueeze(2)
    pose_pred = (p1 * x_in.unsqueeze(1)).sum(3)
    return pose_pred

def compute_symmetry_loss(x):
    start_joints_idx = [11, 8, 5, 2, 10, 7, 4, 1, 23, 21, 19, 17, 14, 22, 20, 18, 16, 13]
    end_joints_idx = [8, 5, 2, 0, 7, 4, 1, 0, 21, 19, 17, 14, 9, 20, 18, 16, 13, 9]
    symm_pairs1 = [0, 1, 2, 3, 8, 9, 10, 11, 12]
    symm_pairs2 = [4, 5, 6, 7, 13, 14, 15, 16, 17]

    bone_length = torch.sqrt(torch.sum(torch.square(x[:, start_joints_idx] - x[:, end_joints_idx]), dim=2))
    loss = nn.MSELoss()
    return loss(bone_length[:, symm_pairs1], bone_length[:, symm_pairs2])


def compute_range_loss(x, range_min, range_max):
    start_joints_idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    end_joints_idx = [0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
    bone_length = torch.sqrt(torch.sum(torch.square(x[:, start_joints_idx] - x[:, end_joints_idx]), dim=2))

    outlier_mask = torch.sign(bone_length - range_min) == torch.sign(bone_length - range_max)
    loss = torch.min(torch.abs(bone_length - range_min), torch.abs(bone_length - range_max)) * outlier_mask
    loss = torch.mean(loss)
    return loss


def compute_angle_loss(x, range_min, range_max):
    start_joints_idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    end_joints_idx = [0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]

    bone_vectors = x[:, start_joints_idx] - x[:, end_joints_idx]
    norm_bone_vectors = bone_vectors / bone_vectors.norm(dim=2, keepdim=True)

    in_vectors = torch.stack([-norm_bone_vectors[:, 0],
                                 -norm_bone_vectors[:, 0],
                                 -norm_bone_vectors[:, 1],
                                 norm_bone_vectors[:, 0],
                                 norm_bone_vectors[:, 3],
                                 norm_bone_vectors[:, 6],
                                 norm_bone_vectors[:, 1],
                                 norm_bone_vectors[:, 4],
                                 norm_bone_vectors[:, 7],
                                 norm_bone_vectors[:, 2],
                                 norm_bone_vectors[:, 5],
                                 norm_bone_vectors[:, 8],
                                 norm_bone_vectors[:, 8],
                                 norm_bone_vectors[:, 8],
                                 norm_bone_vectors[:, 11],
                                 norm_bone_vectors[:, 12],
                                 norm_bone_vectors[:, 15],
                                 norm_bone_vectors[:, 17],
                                 norm_bone_vectors[:, 19],
                                 norm_bone_vectors[:, 13],
                                 norm_bone_vectors[:, 16],
                                 norm_bone_vectors[:, 18],
                                 norm_bone_vectors[:, 20],
                                 -norm_bone_vectors[:, 13],
                                 -norm_bone_vectors[:, 12],
                                 ], dim=1)

    out_vectors = torch.stack([norm_bone_vectors[:, 1],
                                 norm_bone_vectors[:, 2],
                                 norm_bone_vectors[:, 2],
                                 norm_bone_vectors[:, 3],
                                 norm_bone_vectors[:, 6],
                                 norm_bone_vectors[:, 9],
                                 norm_bone_vectors[:, 4],
                                 norm_bone_vectors[:, 7],
                                 norm_bone_vectors[:, 10],
                                 norm_bone_vectors[:, 5],
                                 norm_bone_vectors[:, 8],
                                 norm_bone_vectors[:, 11],
                                 norm_bone_vectors[:, 12],
                                 norm_bone_vectors[:, 13],
                                 norm_bone_vectors[:, 14],
                                 norm_bone_vectors[:, 15],
                                 norm_bone_vectors[:, 17],
                                 norm_bone_vectors[:, 19],
                                 norm_bone_vectors[:, 21],
                                 norm_bone_vectors[:, 16],
                                 norm_bone_vectors[:, 18],
                                 norm_bone_vectors[:, 20],
                                 norm_bone_vectors[:, 22],
                                 norm_bone_vectors[:, 11],
                                 norm_bone_vectors[:, 11],
                                 ], dim=1)

    dot_products = torch.sum(in_vectors * out_vectors, dim=2)
    outlier_mask = torch.sign(dot_products - range_min) == torch.sign(dot_products - range_max)
    loss = torch.min(torch.abs(dot_products - range_min), torch.abs(dot_products - range_max)) * outlier_mask
    loss = torch.mean(loss)
    return loss
