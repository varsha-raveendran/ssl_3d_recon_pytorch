from turtle import pos
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
from src.training.validation import *

import wandb


def train(recon_net,pose_net,device,config,trainloader,valloader, initial_pcl = None):
    # For inference stage optimisation
    if initial_pcl != None:
      initial_pcl = initial_pcl.to(device)

    wandb.init(project='train_ml3d',reinit=True,  config = config)
    print("training model!")


    # Initialize projection modules
    world2cam = World2Cam()
    perspective_transform = PerspectiveTransform()
    get_proj_rgb = RgbContProj()
    get_proj_mask = ContProj()

    ## Define Losses
    img_loss = ImageLoss()
    gcc_loss = GCCLoss()
    pose_loss = PoseLoss()

    optimizer = torch.optim.Adam([
        {
            # TODO: optimizer params and learning rate for model (lr provided in config)
            'params' : recon_net.parameters(),
            'lr': config['learning_rate_recon_net'],
            'weight_decay': 0.00001,
        },
        {
            # TODO: optimizer params and learning rate for latent code (lr provided in config)
            'params': pose_net.parameters(),
            'lr': config['learning_rate_pose_net'],
            'weight_decay': 0.00001,
        }
    ])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000], gamma=0.1, verbose=False)
    ## Setting GPU
    recon_net.to(device)
    pose_net.to(device)

    recon_net.train()
    pose_net.train()

    best_loss = -1*1e+10

    train_loss_running = 0.
    num_epochs = config['max_epochs']
    if(config['use_pretrained']):
      print('Using Pre Trained Model!')
    
    ## Logging metrics for training epochs
    log_total_loss = []
    log_pose_loss = []
    log_recon_loss = []
    log_symm_loss = []

    val_log_total_loss = []
    val_log_pose_loss = []
    val_log_recon_loss = []
    val_log_symm_loss = []

    #wandb.watch(recon_net, log_freq=100,log_graph=False)
    #wandb.watch(pose_net, log_freq=100,log_graph=False)
    for epoch in range(num_epochs):

        if(epoch == 80):
          pose_net.eval()
          config['lambda_3d'] = 0
          config['lambda_symm'] = 0
          config['lambda_pose'] = 0
          config['lambda_ae'] = 10
          config['lambda_ae_mask'] = 10
          config['lambda_ae_pose'] = 0
          config['lambda_mask_pose'] = 0

        train_loss_running = []
        pose_loss_running = []
        recon_loss_running = []
        symm_loss_running = []
        print('**********************************************')
        print('Epoch : ', epoch)
        for i, batch in enumerate(trainloader):
            
            
            # move batch to device
            ShapeNet.move_batch_to_device(batch, device)
            optimizer.zero_grad()

            batch['img_mask'] = 1 - torchvision.transforms.Normalize(batch['img_mask'].mean(), batch['img_mask'].std())(batch['img_mask'])
            # pcl_out = pose_out = img_out = pcl_rgb_out = []
            batch['img_mask'] = (batch['img_mask'] > 0.9).float()
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
                # print("img_out max" , torch.max(temp_img_out[0][0]))
                mask_out.append(get_proj_mask(pcl_out_persp, 64, 64,
                    1024, 0.4,config["device"])) ### Invert mask and image(?)
                # print("mask_out max" , torch.max(mask_out[0][0]))

            temp_img = torch.clone(img_out[0][0])
            temp_mask = torch.clone(mask_out[0][0])

            temp_gray_img = torchvision.transforms.Grayscale()(torch.permute(temp_img,[2,0,1]))
            temp_gray_img = torch.permute(temp_gray_img,[1,2,0])

            temp_gt_mask = torch.clone(batch["img_mask"][0][0])
            
            # print(torch.squeeze(torch.clone(pcl_out[0]),0).T.shape)
            temp_pcl_xyz = torch.permute(torch.squeeze(torch.clone(pcl_out[0]),0).T, [2,0,1])[0].cpu().detach().numpy()
            #temp_pcl_rgb = torch.permute(torch.squeeze(torch.clone(pcl_rgb_out[0]),0).T, [2,0,1])[0].cpu().detach().numpy()
            
    #         point_cloud = wandb.Object3D({'type':'lidar/beta',
    #                                      'points':temp_pcl})
            images = wandb.Image((temp_gray_img.cpu().detach().numpy()*255).astype(np.uint8),
                                caption="Projected Image")
            masks = wandb.Image((temp_mask.cpu().detach().numpy()*255).astype(np.uint8), caption="Projected Mask")
            gt_mask = wandb.Image((temp_gt_mask.cpu().detach().numpy()*255).astype(np.uint8), caption="Ground Truth Mask")
            log_list = [images,masks, gt_mask]
            wandb.log({"image": log_list})
            # print(temp_pcl_xyz.shape)
            wandb.log({"point_cloud_1" : [wandb.Object3D(temp_pcl_xyz)]})


            # print(img_out[1][0].shape)
            # Reconstruct the point cloud from and predict the pose of projected images
            for idx in range(config['n_proj']):
                pcl_xyz, pcl_rgb = recon_net(torch.permute(img_out[idx],[0, 3, 1, 2]).contiguous())
                pcl_out.append(pcl_xyz)
                pcl_rgb_out.append(pcl_rgb)

                pose_out.append(pose_net(torch.permute(img_out[idx],[0, 3, 1, 2]).contiguous()))

            #temp_pcl = torch.permute(torch.squeeze(torch.clone(pcl_out[1]),0).T, [2,0,1])[0].cpu().detach().numpy()
#         temp_pcl2 = torch.squeeze(torch.clone(pcl_out[1]),0).T.cpu().detach().numpy()
            gt_pcl = torch.permute(torch.squeeze(torch.clone(batch['pcl']),0).T, [2,0,1])[0].cpu().detach().numpy()
            wandb.log({"point_cloud_2" : [wandb.Object3D(gt_pcl)]})

            # Define Losses
            # 2D Consistency Loss - L2
            # print('IMAGE LOSS : ',torch.permute(torch.stack(img_out)[0],[0, 3, 1, 2]).contiguous().shape)
            img_ae_loss, _, _ = img_loss(batch['img_rgb'], torch.permute(torch.stack(img_out)[0],[0, 3, 1, 2]).contiguous()
                        , 'l2_sq')
            # print('MASK CHECK : ',batch['img_mask'].shape,torch.stack(mask_out)[0].shape)
            mask_ae_loss, mask_fwd, mask_bwd = img_loss(torch.stack(mask_out)[0], torch.squeeze(batch['img_mask'],1),
                    'bce', affinity_loss=True)
            # print('Mask Loss Check ', mask_ae_loss, mask_fwd, mask_bwd)

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
            total_loss = 0.
            reg_loss = 0.
            recon_loss = (config['lambda_ae']*img_ae_loss) + (config['lambda_3d']*consist_3d_loss)\
                                + (config['lambda_ae_mask']*mask_ae_loss) +\
                                (config['lambda_mask_fwd']*mask_fwd) + (config['lambda_mask_bwd']*mask_bwd)
            if config["use_symmetry_loss"]:
                recon_loss += (config['lambda_symm']*symm_loss)
            pose_loss_val = (config['lambda_ae_pose']*img_ae_loss) + (config['lambda_pose']*pose_loss_pose)\
                            + (config['lambda_mask_pose']*mask_ae_loss)
            
            if(not config['iso']):
                total_loss = recon_loss + pose_loss_val
            else:    
                for idx in range(config['n_proj']):
                  reg_loss += chamfer_distance(torch.permute(initial_pcl,[0,2,1]), torch.permute(pcl_out[idx],[0,2,1]))[0]
                    
                total_loss = (config['lambda_ae']*img_ae_loss) + (config['lambda_3d']*reg_loss)\
                                + (config['lambda_ae_mask']*mask_ae_loss) +\
                                (config['lambda_mask_fwd']*mask_fwd) + (config['lambda_mask_bwd']*mask_bwd) +\
                                      (config['lambda_symm']*symm_loss)

            
            # loss.backward(retain_graph=True)
            # recon_loss.backward(retain_graph=True)
            # pose_loss_val.backward(retain_graph=True)
            if config['iso']:
              total_loss.backward(retain_graph=True)
            else:
              recon_loss.backward(retain_graph=True)
              pose_loss_val.backward(retain_graph=True)  

            optimizer.step()
            # scheduler.step()
            print('Epoch: ', epoch, 'Iteration : ',i)
            print('Loss : ',total_loss.item())
            #wandb.log({'Total Loss': total_loss.item()})
            wandb.log({'Recon_loss': recon_loss.item()})
            wandb.log({'Pose_loss': pose_loss_val.item()})
            # print('recon_loss : ',recon_loss.item())
            # print('pose_loss : ',pose_loss_val.item())

            train_loss_running.append(total_loss.item())
            pose_loss_running.append(pose_loss_val.item())
            recon_loss_running.append(recon_loss.item())
            if config["use_symmetry_loss"]:
                symm_loss_running.append(symm_loss.item())

        ## VALIDATION
        if(not config['iso']):
            val_loss,val_pose_loss,val_recon_loss,val_symm_loss = \
                        validate(recon_net,pose_net,device,config,valloader)
            recon_net.train()
            pose_net.train()

        
            val_log_total_loss.append(val_loss[0])
            val_log_pose_loss.append(val_pose_loss[0])
            val_log_recon_loss.append(val_recon_loss[0])
            if config["use_symmetry_loss"]:
                    val_log_symm_loss.append(val_symm_loss[0])

        ## Logging metrics for each epoch
        log_total_loss.append(sum(train_loss_running)/ len(train_loss_running))
        log_pose_loss.append(sum(pose_loss_running)/ len(pose_loss_running))
        log_recon_loss.append(sum(recon_loss_running)/ len(recon_loss_running))
        if config["use_symmetry_loss"]:
                log_symm_loss.append(sum(symm_loss_running)/ len(symm_loss_running))

        if config["use_symmetry_loss"]:
            log_metrics = pd.DataFrame({'train_total_loss':log_total_loss,
                                        'train_pose_loss':log_pose_loss,
                                        'train_recon_loss':log_recon_loss,
                                        'train_symm_loss': log_symm_loss
                                        })
            if(not config['iso']):
                val_log_metrics = pd.DataFrame({'val_total_loss':val_log_total_loss,
                                            'val_pose_loss':val_log_pose_loss,
                                            'val_recon_loss':val_log_recon_loss,
                                            'val_symm_loss': val_log_symm_loss
                                            })
        else:
            log_metrics = pd.DataFrame({'train_total_loss':log_total_loss,
                                        'train_pose_loss':log_pose_loss,
                                        'train_recon_loss':log_recon_loss
                                        })
            if(not config['iso']):
                val_log_metrics = pd.DataFrame({'val_total_loss':val_log_total_loss,
                                            'val_pose_loss':val_log_pose_loss,
                                            'val_recon_loss':val_log_recon_loss
                                            })
        log_metrics.to_csv(f'src/logs/{config["experiment_name"]}/training_metrics.csv')
        if(not config['iso']):
            val_log_metrics.to_csv(f'src/logs/{config["experiment_name"]}/validation_metrics.csv')

        if(not config['iso']):
          if(epoch ==0 or (val_loss[0])<best_loss):
            print('Saving new model!')
            best_loss = val_loss[0]
            torch.save(recon_net.state_dict(), f'src/runs/{config["experiment_name"]}/recon_model_best_{epoch}.ckpt')
            torch.save(pose_net.state_dict(), f'src/runs/{config["experiment_name"]}/pose_model_best_{epoch}.ckpt')

            # Saving to GDrive
            # torch.save(recon_net.state_dict(), f'../drive/MyDrive/recon_model_best.ckpt')
            # torch.save(pose_net.state_dict(), f'../drive/MyDrive/pose_model_best.ckpt')

        else:
          if(epoch ==0 or (sum(train_loss_running)/ len(train_loss_running))<best_loss):
            print('Saving new model!')
            best_loss = sum(train_loss_running)/ len(train_loss_running)
            torch.save(recon_net.state_dict(), f'src/runs/{config["experiment_name"]}/recon_model_inf_{epoch}.ckpt')
            torch.save(pose_net.state_dict(), f'src/runs/{config["experiment_name"]}/pose_model_inf_{epoch}.ckpt')

            # Saving to GDrive
            # torch.save(recon_net.state_dict(), f'../drive/MyDrive/recon_model_inf.ckpt')
            # torch.save(pose_net.state_dict(), f'../drive/MyDrive/pose_model_inf.ckpt')    

            
            

            



def main(config):

    trainset = ShapeNet('train' if not config['is_overfit'] else 'overfit_10', config['category'], config['n_proj'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)

    valset = ShapeNet('val' if not config['is_overfit'] else 'overfit_10', config['category'], config['n_proj'])
    valloader = torch.utils.data.DataLoader(valset, batch_size=config['batch_size'], shuffle=False)
    
    # device = 
    # declare device
    device = torch.device(config['device'])  ## CHANGE TO CUDA 

    if(config['use_pretrained']):
      recon_net = ReconstructionNet()
      pose_net = PoseNet() ## Need to merge
      ckpt_recon = f'src/runs/{config["experiment_name"]}/recon_model_best.ckpt'
      recon_net.load_state_dict(torch.load(ckpt_recon, map_location='cpu'))
      ckpt_pose = f'src/runs/{config["experiment_name"]}/pose_model_best.ckpt'
      pose_net.load_state_dict(torch.load(ckpt_pose, map_location='cpu'))
    else: 
      recon_net = ReconstructionNet()
      pose_net = PoseNet() ## Need to merge


    #Sets the value of initial prediction for ISO loss
    if(config['iso']):
      recon_net.to(device)
      recon_net.eval()
      for i, batch in enumerate(trainloader):
        initial_pcl = recon_net(batch['img_rgb'].to(device))
      initial_pcl = initial_pcl[0].detach().cpu().numpy()
      train(recon_net,pose_net,device,config,trainloader,valloader, torch.from_numpy(initial_pcl))
    else: 
      train(recon_net,pose_net,device,config,trainloader,valloader)  
