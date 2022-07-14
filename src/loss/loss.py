import numpy as np
from scipy.spatial.distance import cdist as np_cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
# from chamfer_distance import ChamferDistance


class ImageLoss(nn.Module):
    
    def __init__(self):
        super(ImageLoss, self).__init__()

    def grid_dist(grid_h, grid_w):
        '''
        Compute distance between every point in grid to every other point
        '''
        x, y = np.meshgrid(range(grid_h), range(grid_w), indexing='ij')
        grid = np.asarray([[x.flatten()[i],y.flatten()[i]] for i in range(len(x.flatten()))])
        grid_dist = np_cdist(grid,grid)
        grid_dist = np.reshape(grid_dist, [grid_h, grid_w, grid_h, grid_w])
        return grid_dist

    def forward(self, gt, pred, mode='l2_sq',affinity_loss=False):        
        
        '''
        Loss in 2D domain - on mask or rgb images.
        Args:
            gt: (BS, H, W, *); ground truth mask or rgb
            pred: (BS, H, W, *); predicted mask or rgb
            mode: str; loss mode
            affinity_loss: boolean; affinity loss will be added to mask loss if
                                    set to True
        Returns:
            loss: (); averaged loss value
            min_dist: (); averaged forward affinity distance
            min_dist_inv: (); averaged backward affinity distance
        '''
        grid_h, grid_w = 64, 64
        dist_mat = grid_dist(grid_h, grid_w)
        min_dist = min_dist_inv = None
        if mode=='l2_sq':
            loss = (gt - pred)**2
            loss = torch.mean(torch.sum(loss, axis=-1))
        elif mode=='l2':
            loss = (gt - pred)**2
            loss = torch.sum(loss, axis=-1)
            loss = torch.sqrt(loss)
            loss = torch.mean(loss)
        if mode=='l1':
            loss = torch.abs(gt - pred)
            loss = torch.mean(loss)
        elif mode == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(gt, pred)
            loss = torch.mean(loss)
        elif mode == 'bce_prob':
            epsilon = 1e-8
            loss = -gt*torch.log(pred+epsilon) - (1-gt)*torch.log(torch.abs(1-pred-epsilon))
            loss = torch.mean(loss)
        if affinity_loss:
            dist_mat += 1.
            gt_mask = gt #+ (1.-gt)*1e6*tf.ones_like(gt)
            gt_white = torch.unsqueeze(torch.unsqueeze(gt,3),3)
            gt_white = gt_white.repeat(1,1,1,grid_h,grid_w) ### Check tile function

            pred_white = torch.unsqueeze(torch.unsqueeze(pred,3),3)
            pred_white = pred_white.repeat(1,1,1,grid_h,grid_w) ### Check tile function

            gt_white_th = gt_white + (1.-gt_white)*1e6*torch.ones_like(gt_white)
            dist_masked = gt_white_th * dist_mat * pred_white

            pred_mask = (pred_white) + ((1.-pred_white))*1e6*torch.ones_like(pred_white)
            dist_masked_inv = pred_mask * dist_mat * gt_white

            min_dist = torch.mean(torch.amin(dist_masked,(3,4)))
            min_dist_inv = torch.mean(torch.amin(dist_masked_inv,(3,4)))

        return loss,min_dist,min_dist_inv

class GCCLoss(nn.Module):
    
    def __init__(self):
        super(GCCLoss, self).__init__()

    def forward(self, gt, pred, mode='chamfer'):        
        '''
        3D loss: to enforce 3D consistency loss
        Args:
            gt: (BS,N_pts,3); GT point cloud
            pred: (BS,N_pts,3); predicted point cloud
            mode: str; method to calculate loss - 'chamfer' or 'emd'
        Returns:
            loss: (); averaged chamfer/emd loss
        '''
        if mode=='chamfer':
            chamfer_dist = ChamferDistance()
            dist1, dist2, idx1, idx2 = chamfer_dist(gt,pred)
            # _, _, loss = get_chamfer_dist(gt, pred)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
        # elif mode=='emd':
            # loss = get_emd_dist(gt, pred)
        # loss = tf.reduce_mean(loss)
        return loss

class PoseLoss(nn.Module):
    
    def __init__(self):
        super(PoseLoss, self).__init__()

    def forward(self, gt, pred, mode='l1'):        
        '''
        Pose loss: to enforce pose consistency loss
        Args:
            gt: (BS,2); GT pose - azimuth and elevation for each pcl
            pred: (BS,2); predicted pose
            mode: str; method to calculate loss
        Returns:
            loss: (); averaged loss
        '''
        if mode=='l1':
            loss = torch.mean(torch.abs(gt - pred))
        elif mode=='l2_sq':
            loss = (gt - pred)**2
            loss = torch.mean(torch.sum(loss, axis=-1))
        elif mode=='l2':
            loss = (gt - pred)**2
            loss = torch.sum(loss, axis=-1)
            loss = torch.sqrt(loss)
            loss = torch.mean(loss)
        elif mode == 'cosine_dist':
            pred_norm = torch.sqrt(torch.sum(pred**2, axis=-1)+1e-8)
            loss = 1. - (torch.sum(gt*pred, axis=-1)/(pred_norm+1e-8))
        
        return loss