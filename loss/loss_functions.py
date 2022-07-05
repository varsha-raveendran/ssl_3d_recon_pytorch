import torch
import numpy as np
from scipy.spatial.distance import cdist as np_cdist
import torch.nn.functional as F
# from chamfer_distance import ChamferDistance

def test_loss():

    return 42

def grid_dist(grid_h, grid_w):
    '''
    Compute distance between every point in grid to every other point
    '''
    x, y = np.meshgrid(range(grid_h), range(grid_w), indexing='ij')
    grid = np.asarray([[x.flatten()[i],y.flatten()[i]] for i in range(len(x.flatten()))])
    grid_dist = np_cdist(grid,grid)
    grid_dist = np.reshape(grid_dist, [grid_h, grid_w, grid_h, grid_w])
    return grid_dist


def get_img_loss(gt, pred, mode='l2_sq', affinity_loss=False):
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
        # gt_white = torch.tile(gt_white, (1,1,1,grid_h,grid_w))
        gt_white = gt_white.repeat(1,1,1,grid_h,grid_w) ### Check tile function

        pred_white = torch.unsqueeze(torch.unsqueeze(pred,3),3)
        # pred_white = torch.tile(pred_white, (1,1,1,grid_h,grid_w))
        pred_white = pred_white.repeat(1,1,1,grid_h,grid_w) ### Check tile function

        gt_white_th = gt_white + (1.-gt_white)*1e6*torch.ones_like(gt_white)
        dist_masked = gt_white_th * dist_mat * pred_white

        pred_mask = (pred_white) + ((1.-pred_white))*1e6*torch.ones_like(pred_white)
        dist_masked_inv = pred_mask * dist_mat * gt_white

        min_dist = torch.mean(torch.min(dist_masked, axis=[3,4]))
        min_dist_inv = torch.mean(torch.min(dist_masked_inv, axis=[3,4]))

    return loss,min_dist,min_dist_inv


def get_2d_symm_loss(img):
    '''
    Symmetry loss: to enforce symmetry in image along vertical axis in the
    image
    Args:
        img: (BS,H,W,3); input image
    Returns:
        loss: (); averaged symmetry loss, L1
    '''
    W = img.shape[2]
    loss = torch.abs(img[:,:,:W/2,:]-img[:,:,W/2:,:])
    loss = torch.mean(loss)
    return loss


def get_pose_loss(gt, pred, mode='l1'):
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
	# print '\nCosine Distance\n'
    	pred_norm = torch.sqrt(torch.sum(pred**2, axis=-1)+1e-8)
    	loss = 1. - (torch.sum(gt*pred, axis=-1)/(pred_norm+1e-8))
    return loss

def get_3d_loss(gt, pred, mode='chamfer'):
    '''
    3D loss: to enforce 3D consistency loss
    Args:
        gt: (BS,N_pts,3); GT point cloud
        pred: (BS,N_pts,3); predicted point cloud
        mode: str; method to calculate loss - 'chamfer' or 'emd'
    Returns:
        loss: (); averaged chamfer/emd loss
    '''
    if mode=='chamfer' or mode=='emd':
        chamfer_dist = ChamferDistance()
        dist1, dist2, idx1, idx2 = chamfer_dist(gt,pred)
        # _, _, loss = get_chamfer_dist(gt, pred)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
    # elif mode=='emd':
        # loss = get_emd_dist(gt, pred)
    # loss = tf.reduce_mean(loss)
    return loss



# def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
#     # You can comment out this line if you are passing tensors of equal shape
#     # But if you are passing output from UNet or something it will most probably
#     # be with the BATCH x 1 x H x W shape
#     # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

#     SMOOTH = 1e-6
    
#     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
#     return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch


# def get_partseg_loss(gt, pred, loss='ce_logits'):
#     '''
#     Calculate part segmentation loss
#     Args:
#         gt_pcl: (BS,N_pts); GT point cloud
#         pred_pcl: (BS,N_pts,N_cls); predicted point cloud
#         loss: str; type of loss to be used: 'ce_logits' or 'iou'
#     Returns:
#         loss: (BS); averaged loss
#     '''
#     if loss == 'ce_logits':
#     # print '\nBCE Logits Loss\n'
#         loss = F.nll_loss(F.softmax(pred), gt)
#     elif loss == 'iou':
#     # print '\nIoU metric\n'
#         # pred_idx = torch.argmax(pred, axis=3)
#         # loss = tf.metrics.mean_iou(gt, pred_idx, NUM_CLASSES+1) # tuple of (iou, conf_mat)
#         loss = iou_pytorch(gt,pred)
#     return loss


