import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class World2Cam(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    # Camera origin calculation - az,el,d to 3D co-ord
    def forward(self, xyz, az, el, d_min, d_max, batch_size, n_pts=1024):
        d = d_min
        # Calculate translation params
        tx, ty, tz = [0, 0, d]
        rotmat_az=[
		    [torch.ones_like(az),torch.zeros_like(az),torch.zeros_like(az)],
		    [torch.zeros_like(az),torch.cos(az),-torch.sin(az)],
		    [torch.zeros_like(az),torch.sin(az),torch.cos(az)]
		    ]
        rotmat_el=[
		    [torch.cos(el),torch.zeros_like(az), torch.sin(el)],
		    [torch.zeros_like(az),torch.ones_like(az),torch.zeros_like(az)],
		    [-torch.sin(el),torch.zeros_like(az), torch.cos(el)]
		    ]
        rotmat_az = torch.transpose(torch.stack(rotmat_az, 0), [2,0,1])  
        rotmat_el = torch.transpose(torch.stack(rotmat_el, 0), [2,0,1])

        rotmat = torch.matmul(rotmat_el, rotmat_az)

        tr_mat = torch.tile(torch.expand_dims([tx, ty, tz],0), [batch_size,1]) # [B,3]
        tr_mat = torch.expand_dims(tr_mat,2) # [B,3,1]
        tr_mat = torch.transpose(tr_mat, [0,2,1]) # [B,1,3]
        tr_mat = torch.tile(tr_mat,[1,n_pts,1]) # [B,2048,3]

        xyz_out = torch.matmul(rotmat,torch.transpose((xyz),[0,2,1])) - torch.transpose(tr_mat,[0,2,1])
        return torch.transpose(xyz_out,[0,2,1])


model = World2Cam().cuda()
      