import numpy as np
import torch
from pathlib import Path
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import corresponding_points_alignment

def evaluate_pred_pcl(item, config):
    
    device = torch.device(config['device'])
    pred_pcl = np.load(f'output/{config["category"]}/{item["name"]}.npy', allow_pickle=True)
    print(pred_pcl.shape)
    
    pred_pcl = torch.from_numpy(pred_pcl).permute(1,0)
    
    gt_pcl = item["pcl"].permute(1,0)
    print(gt_pcl.shape)
    
    # get the transformation 
    R, T, _ = corresponding_points_alignment(pred_pcl.unsqueeze(0), gt_pcl.unsqueeze(0))
    pcl_rot = R.squeeze(0) @ torch.permute(pred_pcl, (1,0))
    loss = chamfer_distance(pcl_rot.permute(1,0).unsqueeze(0),gt_pcl.unsqueeze(0))
    print("loss" , loss )


def main(config):
    testset = ShapeNet('test', config['category'], config['n_proj'])

    for item in testset:
        evaluate_pred_pcl(item, config)