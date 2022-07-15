import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class World2Cam(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    # Camera origin calculation - az,el,d to 3D co-ord
    def forward(self, xyz, az, el, d_min, d_max, batch_size, device, n_pts=1024):
        device = torch.device(device)
        d = d_min
        # Calculate translation params
        #tx, ty, tz = [0, 0, d]
        tx = torch.tensor(0)
        ty = torch.tensor(0)
        tz = torch.tensor(d)
        # rotmat_az=[
		#     [torch.ones_like(az),torch.zeros_like(az),torch.zeros_like(az)],
		#     [torch.zeros_like(az),torch.cos(az),-torch.sin(az)],
		#     [torch.zeros_like(az),torch.sin(az),torch.cos(az)]
		#     ]
        # rotmat_el=[
		#     [torch.cos(el),torch.zeros_like(az), torch.sin(el)],
		#     [torch.zeros_like(az),torch.ones_like(az),torch.zeros_like(az)],
		#     [-torch.sin(el),torch.zeros_like(az), torch.cos(el)]
		#     ]
        # rotmat_az = torch.stack(rotmat_az)
        # rotmat_el = torch.stack(rotmat_el)    

        rotmat_az=[
		    torch.stack([torch.ones_like(az),torch.zeros_like(az),torch.zeros_like(az)]),
		    torch.stack([torch.zeros_like(az),torch.cos(az),-torch.sin(az)]),
		    torch.stack([torch.zeros_like(az),torch.sin(az),torch.cos(az)])
		    ]
        rotmat_el=[
		    torch.stack([torch.cos(el),torch.zeros_like(az), torch.sin(el)]),
		    torch.stack([torch.zeros_like(az),torch.ones_like(az),torch.zeros_like(az)]),
		    torch.stack([-torch.sin(el),torch.zeros_like(az), torch.cos(el)])
		    ]
        
        # For batch - uncomment once debugging done
        rotmat_az = torch.permute(torch.stack(rotmat_az, 0), (2,0,1))  
        rotmat_el = torch.permute(torch.stack(rotmat_el, 0), [2,0,1])
        # For debugging - comment once debugging done
        # rotmat_az = torch.stack(rotmat_az, 0)
        # rotmat_el = torch.stack(rotmat_el, 0)

        rotmat = torch.matmul(rotmat_el, rotmat_az)

        tr_mat = torch.tile(torch.unsqueeze(torch.stack([tx, ty, tz]),0), [batch_size,1]) # [B,3]
        tr_mat = torch.unsqueeze(tr_mat,2) # [B,3,1]
        tr_mat = torch.permute(tr_mat, [0,2,1]) # [B,1,3]
        tr_mat = torch.tile(tr_mat,[1,n_pts,1]) # [B,2048,3]
        rotmat = rotmat.type(torch.FloatTensor).to(device)
        #xyz_out = torch.matmul(rotmat,torch.permute((xyz),[0,2,1])) - torch.permute(tr_mat,[0,2,1])
        xyz_out = torch.matmul(rotmat,xyz) - torch.permute(tr_mat,[0,2,1]).to(device)
        return torch.permute(xyz_out,[0,2,1])

class PerspectiveTransform(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, input, batch_size, device):
        device = torch.device(device)
        K = np.array([
	    [120., 0., -32.],
	    [0., 120., -32.],
	    [0., 0., 1.]]).astype(np.float64)
        K = np.expand_dims(K, 0)
        K = np.tile(K, [batch_size,1,1])
        
        #Convert np matrix to tensor
        K = torch.from_numpy(K).type(torch.FloatTensor).to(device)
        
        
        xyz_out = torch.matmul(K, torch.permute(input, [0,2,1]))
        xy_out = xyz_out[:,:2]/abs(torch.unsqueeze(input[:,:,2],1))
        xyz_out = torch.concat([xy_out, abs(xyz_out[:,2:])],dim=1)

        return torch.permute(xyz_out, [0,2,1])

class RgbContProj(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, pcl, feat, N_pts, grid_h, grid_w, well_radius=1., beta=100, mode='rgb', device="cuda"):
        '''
        2D Projection of any general feature of 3D point cloud
        Args:
        pcl: float, (N_batch,N_Pts,3); input point cloud
            values assumed to be in (-1,1)
        feat: float, (N_batch, N_Pts, N_cls)
        N_pts: int, ()
            Number of points in PCL
        grid_h, grid_w: int, ();
            output depth map height and width
        well_radius: radius of depth well beyond which to mask out probabilities
        mode: str, Choose between ['rgb','partseg']
        Returns:
        proj_feat: float, (N_batch,H,W,N_cls+1)
            output feature map including background label at position 0
        prob: probablility of point being projected at each pixel
            (N_batch,N_pts,grid_h,grid_w)
        mask: bool, (BS,H,W)
                mask of projection
        '''
        device = torch.device(device)
        add_depth_range = torch.tensor([0,0,1], dtype=torch.float32).to(device) # z dim goes from [-1,1] to [0,2]. needed for getting the correct probabilities.
        depth_val = self.get_depth(pcl+add_depth_range, grid_h, grid_w, N_pts, device, well_radius) # (BS,N_PTS,H,W)
        prob = self.get_proj_prob_exp(depth_val, beta) # (BS,N_PTS,H,W)
        # Mask out the regions where no point is projected
        mask = torch.logical_not(torch.eq(10.*torch.ones_like(depth_val).type(torch.float32), depth_val)) # (BS,N_Pts,H,W)
        #mask = torch.cast(mask, torch.float32)
        mask = mask.type(torch.FloatTensor).to(device)
        prob = prob*mask
        # Normalize probabilities
        prob = prob/(torch.sum(prob, dim=1, keepdim=True) + 1e-8)
        # Expectation of feature values
        proj_feat = torch.unsqueeze(prob, dim=-1) * torch.permute(torch.unsqueeze(torch.unsqueeze(feat.type(torch.float32),
        dim=2), dim=2), [0,4,2,3,1]) # (BS,N_pts,H,W,N_cls)
        proj_feat = torch.sum(proj_feat, dim=1) # (BS,H,W,N_cls) --> one-hot
        # mask out background i.e. regions where all point contributions sum to 0
        BS,H,W,_ = [int(d) for d in proj_feat.shape]
        mask = torch.sum(mask, dim=1) # (BS,H,W)
        
        if mode == 'partseg':
        # Insert background label at position 0
            #mask = torch.cast(torch.equal(torch.zeros_like(mask), mask), torch.float32) # (BS,H,W)
            mask = torch.eq(torch.zeros_like(mask), mask).type(torch.FloatTensor) # (BS,H,W)
            bgnd_lbl = torch.ones(shape=(BS,H,W,1)) * torch.unsqueeze(mask,dim=-1) #(BS,H,W,1)
            proj_feat = torch.concat([bgnd_lbl,proj_feat], dim=-1) #(BS,H,W,N_cls+1)
        elif mode == 'rgb' or mode =='normals':
            # remove color/normals from background regions
            #mask = torch.cast(torch.logical_not(torch.equal(torch.zeros_like(mask), mask)), torch.float32) # (BS,H,W)
            mask = torch.logical_not(torch.eq(torch.zeros_like(mask), mask)).type(torch.FloatTensor) # (BS,H,W)
            proj_feat = proj_feat * torch.unsqueeze(mask,dim=-1).to(device)
        
        return proj_feat, prob, mask            
        
    def get_depth(self, pcl, grid_h, grid_w, N_pts, device, well_radius=0.5,):
        '''
        Well function for obtaining depth of every 3D input point at every 2D pixel
        Args:
            pcl: float, (N_batch,N_Pts,3); input point cloud values assumed to be in (0,2)
            grid_h, grid_w: int, (); output depth map height and width
        Returns:
        depth: float, (N_batch,N_Pts,H,W); output depth
        '''
        x, y, z = torch.chunk(pcl, 3, dim=2)
        # x,y,z = pcl[:, :, 0], pcl[:, :, 1], pcl[:, :, 2]
        #pcl_norm = torch.concat([x, y, z], 2)
        #pcl_xy = torch.concat([x,y], 2)
        pcl_xy = pcl[:,:,0:2]
        out_grid = torch.meshgrid(torch.arange(grid_h).to(device), torch.arange(grid_w).to(device), indexing='ij')
        # out_grid = [torch.to_float(out_grid[0]), torch.to_float(out_grid[1])]
        out_grid = [out_grid[0].type(torch.FloatTensor), out_grid[1].type(torch.FloatTensor)]
        grid_z = torch.unsqueeze(torch.zeros_like(out_grid[0]), dim=2).to(device) # (H,W,1)
        # grid_xyz = torch.concat([torch.stack(out_grid, dim=2), grid_z], dim=2)  # (H,W,3)
        grid_xy = torch.stack(out_grid, dim=2).to(device)    # (H,W,2)
        grid_diff = torch.unsqueeze(torch.unsqueeze(pcl_xy, dim=2), dim=2) - grid_xy    # (BS,N_PTS,H,W,2)
        grid_val = self.apply_ideal_kernel_depth(grid_diff, N_pts, device, well_radius)    # (BS,N_PTS,H,W,2)
        grid_val = grid_val[:,:,:,:,0]*grid_val[:,:,:,:,1]*torch.unsqueeze(z,3)  # (BS,N_PTS,H,W)
        # grid_val = grid_val[:,:,:,:,0]*grid_val[:,:,:,:,1]*z
        depth = torch.clamp(grid_val,0.,10.)
        return depth

    def apply_ideal_kernel_depth(self, x, N_pts, device, well_radius=0.5,):
        out = torch.where(torch.abs(x)<=well_radius, torch.ones_like(x).to(device), 10*torch.ones_like(x).to(device))
        return out    

    def get_proj_prob_exp(self, d, beta=5., N_pts=1024, ideal=False):
        '''
        Probability of a point being projected at each pixel of the projection
        map
        Args:
        d: depth value of each point when projected at each pixel of projection
        (N_batch,N_pts,grid_h,grid_w). This value is between 0 and 10. For
        points that are within 0.5 distance from grid point, it is max(0,z),
        for the rest, it is min(10,10z).
        Returns:
        prob: probablility of point being projected at each pixel
            float, (N_batch,N_pts,grid_h,grid_w)
        '''
        d_inv = 1. / (d+1e-5)
        if ideal:
        # ideal projection probabilities - ptob=1 for min depth, 0 for rest
            prob = torch.transpose(F.one_hot(torch.argmax(d_inv, dim=1), N_pts), [0,3,1,2])
        else:
            prob = F.softmax(d_inv*beta, dim=1) # for every pixel, apply softmax across all points
        return prob    

class ContProj(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
    def forward(self, pcl, grid_h, grid_w, N_pts, sigma_sq=0.5, device="cuda"):
        '''
        Continuous approximation of Orthographic projection of point cloud
        to obtain Silhouette
        Args:
            pcl: float, (N_batch,N_Pts,3); input point cloud
                    values assumed to be in (-1,1)
            grid_h, grid_w: int, ();
                    output depth map height and width
            N_pts: int, ()
                    number of points in point cloud
            sigma_sq: float, ()
                    value of sigma_squared in projection kernel
        Returns:
            grid_val: float, (N_batch,H,W); output silhouette
        '''    
        device = torch.device(device)
        x, y, z = torch.chunk(pcl, 3, dim=2)
        pcl_norm = torch.concat([x, y, z], 2)
        pcl_xy = torch.concat([x,y], 2)
        out_grid = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
        #out_grid = [tf.to_float(out_grid[0]), tf.to_float(out_grid[1])]
        out_grid = [out_grid[0].type(torch.FloatTensor),out_grid[1].type(torch.FloatTensor)]
        grid_z = torch.unsqueeze(torch.zeros_like(out_grid[0]), dim=2) # (H,W,1)
        grid_xyz = torch.concat([torch.stack(out_grid, dim=2), grid_z], dim=2)  # (H,W,3)
        grid_xy = torch.stack(out_grid, dim=2).to(device)                # (H,W,2)
        grid_diff = torch.unsqueeze(torch.unsqueeze(pcl_xy, dim=2), dim=2) - grid_xy # (BS,N_PTS,H,W,2)
        grid_val = self.apply_kernel(grid_diff, sigma_sq)    # (BS,N_PTS,H,W,2)
        grid_val = grid_val[:,:,:,:,0]*grid_val[:,:,:,:,1]  # (BS,N_PTS,H,W)
        grid_val = torch.sum(grid_val, dim=1)          # (BS,H,W)
        grid_val = torch.tanh(grid_val)
        return grid_val

    def apply_kernel(self, x, sigma_sq=0.5):
        out = (torch.exp(-(x**2)/(2.*sigma_sq)))
        return out    


      