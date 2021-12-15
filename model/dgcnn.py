import torch
import torch.nn as nn
from model.da_utils import compute_symmetry_loss, compute_range_loss, compute_angle_loss, get_pose_pred


##### DGCNN, adapted from https://github.com/AnTao97/dgcnn.pytorch #####
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=40, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    # if idx is None:
    #    if dim9 == False:
    #        idx = knn(x, k=k)   # (batch_size, num_points, k)
    #    else:
    #        idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNN(nn.Module):
    def __init__(self, cfg, k=40, emb_dims=1024):
        super(DGCNN, self).__init__()
        input_channels = 3
        output_channels = cfg.OUTPUT.NUM_JOINTS

        base = 64
        self.k = k
        self.emb_dims = emb_dims

        self.bn1 = nn.BatchNorm2d(base)
        self.bn2 = nn.BatchNorm2d(base)
        self.bn3 = nn.BatchNorm2d(base)
        self.bn4 = nn.BatchNorm2d(base)
        self.bn5 = nn.BatchNorm2d(base)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(8 * base)
        self.bn8 = nn.BatchNorm1d(4 * base)

        self.conv1 = nn.Sequential(nn.Conv2d(2 * input_channels, base, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(base, base, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(base * 2, base, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(base, base, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(base * 2, base, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(base * 3, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(base * 19, base * 8, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(base * 8, base * 4, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Conv1d(base * 4, output_channels, kernel_size=1, bias=True)

        # domain adaptation stuff
        self.da_method = cfg.DA.METHOD
        if 'anatomical_constraint' in self.da_method:
            self.symmetry_factor = cfg.DA.ANATOMY_SYMMETRY
            self.range_factor = cfg.DA.ANATOMY_RANGE
            self.angle_factor = cfg.DA.ANATOMY_ANGLE

            self.bone_upper = torch.tensor(cfg.DA.ANATOMY_BONE_LENGTH_UPPER, device=cfg.MODEL.DEVICE).float().unsqueeze(0)
            self.bone_lower = torch.tensor(cfg.DA.ANATOMY_BONE_LENGTH_LOWER, device=cfg.MODEL.DEVICE).float().unsqueeze(0)
            self.angle_upper = torch.tensor(cfg.DA.ANATOMY_ANGLE_UPPER, device=cfg.MODEL.DEVICE).float().unsqueeze(0)
            self.angle_lower = torch.tensor(cfg.DA.ANATOMY_ANGLE_LOWER, device=cfg.MODEL.DEVICE).float().unsqueeze(0)

    def forward(self, x_s, x_t=None):
        if self.da_method in ['', 'oracle'] or not self.training:
            x = x_s
            batch_size = x.size(0)
            num_points = x.size(2)

            idx = knn(x.view(batch_size, -1, num_points), k=self.k)

            x = get_graph_feature(x, k=self.k, idx=idx)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
            x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
            x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
            x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

            x = get_graph_feature(x1, k=self.k,
                                  idx=idx)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
            x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
            x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
            x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

            x = get_graph_feature(x2, k=self.k,
                                  idx=idx)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
            x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
            x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

            x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

            x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
            x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

            x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
            x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

            x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
            x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
            x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

            return x, torch.tensor([0.], device=x.device)

        elif 'anatomical_constraint' in self.da_method:
            B_s = x_s.size(0)
            B_t = x_t.size(0)
            x_t_in = x_t.clone()
            x = torch.cat((x_s, x_t), dim=0)

            # joint feature computation for source and target clouds
            batch_size = x.size(0)
            num_points = x.size(2)

            idx = knn(x.view(batch_size, -1, num_points), k=self.k)

            x = get_graph_feature(x, k=self.k, idx=idx)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
            x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
            x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
            x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

            x = get_graph_feature(x1, k=self.k,
                                  idx=idx)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
            x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
            x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
            x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

            x = get_graph_feature(x2, k=self.k,
                                  idx=idx)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
            x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
            x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

            x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

            x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
            x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

            x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
            x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

            x_s = x[:B_s]
            x_s = self.conv7(x_s)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
            x_s = self.conv8(x_s)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
            x_s = self.conv9(x_s)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

            # parameters of network heads are not optimized on target data --> set requires_grad to False
            for layer in [self.conv7, self.conv8, self.conv9]:
                for param in layer.parameters():
                    param.requires_grad = False
            x_t = x[B_s:]
            x_t = self.conv7(x_t)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
            x_t = self.conv8(x_t)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
            x_t = self.conv9(x_t)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
            for layer in [self.conv7, self.conv8, self.conv9]:
                for param in layer.parameters():
                    param.requires_grad = True

            # compute anatomical loss on target predictions
            da_loss = 0.
            x_t = get_pose_pred(x_t_in, x_t)
            if self.symmetry_factor > 0:
                da_loss += self.symmetry_factor * compute_symmetry_loss(x_t)
            if self.range_factor > 0:
                da_loss += self.range_factor * compute_range_loss(x_t, range_min=self.bone_lower, range_max=self.bone_upper)
            if self.angle_factor > 0:
                da_loss += self.angle_factor * compute_angle_loss(x_t, range_min=self.angle_lower, range_max=self.angle_upper)

            return x_s, da_loss
