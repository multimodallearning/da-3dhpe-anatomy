import os
import numpy as np
import torch.utils.data


class SLPDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, domain, phase, split, cover_conds=[]):
        positions = cfg.SLP_DATASET.POSITION
        if positions == 'all':
            positions = ['supine', 'left', 'right']
        elif positions == 'lateral':
            positions = ['left', 'right']
        else:
            positions = [positions]

        if split == 'train':
            # leave out subject 7 due to calibration issue faced by Clever et al.
            subjects = np.concatenate((np.arange(1, 7), np.arange(8, 71)))
            subjects = np.random.default_rng(12345).permutation(subjects)
            if domain == 'source':
                subjects = subjects[:29]
            elif domain == 'target':
                subjects_cov1 = subjects[29:49]
                subjects_cov2 = subjects[49:]
            else:
                raise ValueError()
        elif split == 'val':
            subjects = np.arange(71, 81)
        elif split == 'test':
            subjects = np.arange(81, 103)
        else:
            raise ValueError('The specified split {} is not a valid data split.'.format(split))

        # build data list
        self.data_list = []
        if split == 'train' and domain == 'target':
            for subj in subjects_cov1:
                for pos in positions:
                    if pos == 'supine':
                        pose_no_range = [1, 16]
                    elif pos == 'left':
                        pose_no_range = [16, 31]
                    elif pos == 'right':
                        pose_no_range = [31, 46]
                    else:
                        raise ValueError
                    for pose_no in range(pose_no_range[0], pose_no_range[1]):
                        item = {'subj_no': subj,
                                'pos': pos,
                                'cover_cond': 'cover1',
                                'pose_no': pose_no
                                }
                        self.data_list.append(item)
            for subj in subjects_cov2:
                for pos in positions:
                    if pos == 'supine':
                        pose_no_range = [1, 16]
                    elif pos == 'left':
                        pose_no_range = [16, 31]
                    elif pos == 'right':
                        pose_no_range = [31, 46]
                    else:
                        raise ValueError
                    for pose_no in range(pose_no_range[0], pose_no_range[1]):
                        item = {'subj_no': subj,
                                'pos': pos,
                                'cover_cond': 'cover2',
                                'pose_no': pose_no
                                }
                        self.data_list.append(item)

        else:
            if cover_conds == []:
                if domain == 'source':
                    cover_conds = ['uncover']
                elif domain == 'target':
                    cover_conds = ['cover1', 'cover2']
                else:
                    raise ValueError()
            for subj in subjects:
                for pos in positions:
                    if pos == 'supine':
                        pose_no_range = [1, 16]
                    elif pos == 'left':
                        pose_no_range = [16, 31]
                    elif pos == 'right':
                        pose_no_range = [31, 46]
                    else:
                        raise ValueError
                    for cover_cond in cover_conds:
                        for pose_no in range(pose_no_range[0], pose_no_range[1]):
                            item = {'subj_no': subj,
                                    'pos': pos,
                                    'cover_cond': cover_cond,
                                    'pose_no': pose_no
                                    }
                            self.data_list.append(item)

        self.root = cfg.SLP_DATASET.ROOT
        self.input_template = '3d_data_{}_{}/danaLab_{:05d}_{}_{:03d}_bed_pcd.npy'
        self.target_template = 'gt_joints_3d/subject_{:03d}_pose_{:02d}_gt_joints.npy'

        self.is_train = True if phase == 'train' else False

        self.num_points = cfg.INPUT.NUM_POINTS
        self.rotation = cfg.INPUT.ROT_DEGREE
        self.translation = cfg.INPUT.TRANSLATION

    def __getitem__(self, idx):
        # create paths of input pcd and gt joints
        item = self.data_list[idx]
        pcd_path = self.input_template.format(item['pos'],
                                              item['cover_cond'],
                                              item['subj_no'],
                                              item['cover_cond'],
                                              item['pose_no'])
        joints_path = self.target_template.format(item['subj_no'], item['pose_no'])

        # load pcd
        pcd = np.float32(np.load(os.path.join(self.root, pcd_path)))
        mean = np.mean(pcd, axis=0)
        pcd -= mean

        # load joints
        joints = np.float32(np.load(os.path.join(self.root, joints_path)))
        joints -= mean

        # data augmentation:
        if self.is_train and self.rotation > 0.:
            theta = np.deg2rad(np.random.uniform(-self.rotation, self.rotation))
            rotation_matrix = np.float32(np.array([[np.cos(theta), -np.sin(theta), 0],
                                                   [np.sin(theta), np.cos(theta), 0],
                                                   [0., 0., 1.], ]))
            pcd = np.dot(pcd, rotation_matrix)
            joints = np.dot(joints, rotation_matrix)

        if self.is_train and self.translation > 0.:
            transl = np.random.uniform(-1., 1., (1, 3)) * self.translation
            pcd += transl
            joints += transl

        # subsample points of point cloud
        if self.num_points > 0:
            pts_size = pcd.shape[0]

            if pts_size >= self.num_points:
                if self.is_train:
                    permutation = np.random.default_rng().permutation(pts_size)
                else:
                    permutation = np.random.default_rng(12345).permutation(pts_size)
                pcd = pcd[permutation, :]
                pcd = pcd[:self.num_points, :]
            else:
                if self.is_train:
                    pts_idx = np.random.choice(pts_size, self.num_points, replace=True)
                else:
                    pts_idx = np.random.default_rng(12345).choice(pts_size, self.num_points, replace=True)
                pcd = pcd[pts_idx, :]
        pcd = pcd.T

        return pcd, joints, idx

    def __len__(self):
        return len(self.data_list)
