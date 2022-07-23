import numpy as np
import pandas as pd
import torch
from pathlib import Path
import torchvision
from pytorch3d.loss import chamfer_distance

from src.data.shapenet import ShapeNet
from src.network_architecture.recon_model import ReconstructionNet
from src.network_architecture.pose_net import PoseNet
from src.renderer.projection import World2Cam, PerspectiveTransform, RgbContProj, ContProj
from src.loss.loss import ImageLoss,GCCLoss,PoseLoss




def validate(recon_net,pose_net,device,config,valloader):

    print("Validating model!")

    # Initialize projection modules
    world2cam = World2Cam()
    perspective_transform = PerspectiveTransform()
    get_proj_rgb = RgbContProj()
    get_proj_mask = ContProj()

    ## Define Losses
    img_loss = ImageLoss()
    gcc_loss = GCCLoss()
    pose_loss = PoseLoss()

    recon_net.eval()
    pose_net.eval()

    best_loss = -1*1e+10

    train_loss_running = 0.
    num_epochs = config['max_epochs']
    if(config['use_pretrained']):
      print('Using Pre Trained Model!')
      num_epochs = 200
    
    ## Logging metrics for training epochs
    log_total_loss = []
    log_pose_loss = []
    log_recon_loss = []
    log_symm_loss = []

    for epoch in range(1):
        train_loss_running = []
        pose_loss_running = []
        recon_loss_running = []
        symm_loss_running = []
        for i, batch in enumerate(valloader):
            
            
            # move batch to device
            ShapeNet.move_batch_to_device(batch, device)
            # optimizer.zero_grad()

            batch['img_mask'] = 1 - torchvision.transforms.Normalize(batch['img_mask'].mean(), batch['img_mask'].std())(batch['img_mask'])
            # pcl_out = pose_out = img_out = pcl_rgb_out = []
            pcl_out_rot = []
            pcl_out_persp = []
            mask_out = []
            pcl_out = []
            pose_out = []
            img_out = []
            pcl_rgb_out = []

            ## GET RECONSTRUCTION FROM INPUT IMAGE
            # print('OG IMAGE : ',batch['img_rgb'].shape)
            pcl_xyz,pcl_rgb = recon_net(batch['img_rgb'])
            pcl_out.append(pcl_xyz)
            pcl_rgb_out.append(pcl_rgb)

            ## Currently hard coded number of random poses to 2
            pose_ip = batch['random_pose'] ## Batch Size x 2 x 2

            ## USE POSENET TO GET POSE PREDICTIONS (B,2)
            pose_out.append(pose_net(batch['img_rgb']))
            # v0,v1,v^
            pose_all = torch.concat([torch.unsqueeze(pose_out[0], axis=1), pose_ip], axis=1)


            ## USE THE PORJECTION MODULE TO GET PROJECTIONS FROM PC1 AND POSES
            for idx in range(config['n_proj']):

                # print("PointClouds:", pcl_out[0].shape)
                pcl_out_rot = world2cam(pcl_out[0], pose_all[:, idx, 0],
                    pose_all[:,idx,1], 2., 2., batch['img_rgb'].shape[0],config["device"])
                pcl_out_persp = perspective_transform(pcl_out_rot,
                    batch['img_rgb'].shape[0],config["device"])    
                temp_img_out = get_proj_rgb(pcl_out_persp, pcl_rgb_out[0], 1024,
                    64, 64,1., 100, 'rgb',config["device"])
                # print('Proje img out : ',temp_img_out[0].shape)
                img_out.append(temp_img_out[0])
                mask_out.append(get_proj_mask(pcl_out_persp, 64, 64,
                    1024, 0.4,config["device"])) ### Invert mask and image(?)

    #         temp_img = torch.clone(img_out[0][0])
    #         temp_mask = torch.clone(mask_out[0][0])

    #         temp_gray_img = torchvision.transforms.Grayscale()(torch.permute(temp_img,[2,0,1]))
    #         temp_gray_img = torch.permute(temp_gray_img,[1,2,0])

    #         temp_gt_mask = torch.clone(batch["img_mask"][0][0])
            
    #         # print(torch.squeeze(torch.clone(pcl_out[0]),0).T.shape)
    #         temp_pcl_xyz = torch.permute(torch.squeeze(torch.clone(pcl_out[0]),0).T, [2,0,1])[0].cpu().detach().numpy()
    #         temp_pcl_rgb = torch.permute(torch.squeeze(torch.clone(pcl_rgb_out[0]),0).T, [2,0,1])[0].cpu().detach().numpy()
            
    # #         point_cloud = wandb.Object3D({'type':'lidar/beta',
    # #                                      'points':temp_pcl})
    #         images = wandb.Image((temp_gray_img.cpu().detach().numpy()*255).astype(np.uint8),
    #                             caption="Projected Image")
    #         masks = wandb.Image((temp_mask.cpu().detach().numpy()*255).astype(np.uint8), caption="Projected Mask")
    #         gt_mask = wandb.Image((temp_gt_mask.cpu().detach().numpy()*255).astype(np.uint8), caption="Ground Truth Mask")
    #         log_list = [images,masks, gt_mask]
    #         wandb.log({"image": log_list})
    #         # print(temp_pcl_xyz.shape)
    #         wandb.log({"point_cloud_1" : [wandb.Object3D(temp_pcl_xyz),wandb.Object3D(temp_pcl_rgb)]})


            # print(img_out[1][0].shape)
            # Reconstruct the point cloud from and predict the pose of projected images
            for idx in range(config['n_proj']):
                pcl_xyz, pcl_rgb = recon_net(torch.permute(img_out[idx],[0, 3, 1, 2]).contiguous())
                pcl_out.append(pcl_xyz)
                pcl_rgb_out.append(pcl_rgb)

                pose_out.append(pose_net(torch.permute(img_out[idx],[0, 3, 1, 2]).contiguous()))

#             temp_pcl = torch.permute(torch.squeeze(torch.clone(pcl_out[1]),0).T, [2,0,1])[0].cpu().detach().numpy()
# #         temp_pcl2 = torch.squeeze(torch.clone(pcl_out[1]),0).T.cpu().detach().numpy()
#             gt_pcl = torch.permute(torch.squeeze(torch.clone(batch['pcl']),0).T, [2,0,1])[0].cpu().detach().numpy()
#             wandb.log({"point_cloud_2" : [wandb.Object3D(temp_pcl),wandb.Object3D(gt_pcl)]})

            # Define Losses
            # 2D Consistency Loss - L2
            # print('IMAGE LOSS : ',torch.permute(torch.stack(img_out)[0],[0, 3, 1, 2]).contiguous().shape)
            img_ae_loss, _, _ = img_loss(batch['img_rgb'], torch.permute(torch.stack(img_out)[0],[0, 3, 1, 2]).contiguous()
                        , 'l2_sq')
            # print('MASK CHECK : ',batch['img_mask'].shape,torch.stack(mask_out)[0].shape)
            mask_ae_loss, mask_fwd, mask_bwd = img_loss(torch.squeeze(batch['img_mask'],1), torch.stack(mask_out)[0],
                    'bce', affinity_loss=True)
            # print('Mask Loss Check ')

            # Pose Loss
            pose_all = torch.permute(pose_all,[1,0,2])
            pose_loss_pose = 0
            for idx in range(config['n_proj']):
              pose_loss_pose += pose_loss(pose_all[idx], torch.stack(pose_out)[idx+1], 'l1')
            pose_loss_pose /= config['n_proj']

            # 3D Consistency Loss
            consist_3d_loss = 0.
            for idx in range(config['n_proj']):
                consist_3d_loss += chamfer_distance(torch.permute(pcl_out[idx],[0,2,1]),torch.permute(pcl_out[0],[0,2,1]))[0]

            # consist_3d_loss = 0
            ##TODO
            # Symmetry loss - assumes symmetry of point cloud about z-axis
            # # Helps obtaining output aligned along z-axis
            pcl_y_pos = (pcl_out[0][:,:,1:2]>0).type(torch.FloatTensor).to(device)
            pcl_y_neg = (pcl_out[0][:,:,1:2]<0).type(torch.FloatTensor).to(device)
            pcl_pos = pcl_y_pos*torch.concat([pcl_out[0][:,:,:1], torch.abs(pcl_out[0][:,:,1:2]),
                    pcl_out[0][:,:,2:3]], -1)
            pcl_neg = pcl_y_neg*torch.concat([pcl_out[0][:,:,:1], torch.abs(pcl_out[0][:,:,1:2]),
                    pcl_out[0][:,:,2:3]], -1)
            symm_loss = chamfer_distance(torch.permute(pcl_pos,[0,2,1]), torch.permute(pcl_neg,[0,2,1]))[0]

            # Total Loss
            # loss = (config['lambda_ae']*img_ae_loss) + (config['lambda_3d']*consist_3d_loss) +\
            #         (config['lambda_pose']*pose_loss_pose)
            recon_loss = (config['lambda_ae']*img_ae_loss) + (config['lambda_3d']*consist_3d_loss)\
                            + (config['lambda_ae_mask']*mask_ae_loss) +\
                            (config['lambda_mask_fwd']*mask_fwd) + (config['lambda_mask_bwd']*mask_bwd)
            if config["use_symmetry_loss"]:
                recon_loss += (config['lambda_symm']*symm_loss)
            pose_loss_val = (config['lambda_ae_pose']*img_ae_loss) + (config['lambda_pose']*pose_loss_pose)\
                            + (config['lambda_mask_pose']*mask_ae_loss)

            total_loss = recon_loss + pose_loss_val

            
            print(' Validation Loss : ',total_loss.item())

            train_loss_running.append(total_loss.item())
            pose_loss_running.append(pose_loss_val.item())
            recon_loss_running.append(recon_loss.item())
            if config["use_symmetry_loss"]:
                symm_loss_running.append(symm_loss.item())
        
        log_total_loss.append(sum(train_loss_running)/ len(train_loss_running))
        log_pose_loss.append(sum(pose_loss_running)/ len(pose_loss_running))
        log_recon_loss.append(sum(recon_loss_running)/ len(recon_loss_running))
        if config["use_symmetry_loss"]:
                log_symm_loss.append(sum(symm_loss_running)/ len(symm_loss_running))
        
        return (log_total_loss,log_pose_loss,log_recon_loss,log_symm_loss)
