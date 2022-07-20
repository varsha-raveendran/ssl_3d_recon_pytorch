from turtle import pos
import numpy as np
import torch
from pathlib import Path
import torchvision
from pytorch3d.loss import chamfer_distance

from src.data.shapenet import ShapeNet
from src.network_architecture.recon_model import ReconstructionNet
from src.network_architecture.pose_net import PoseNet
from src.renderer.projection import World2Cam, PerspectiveTransform, RgbContProj, ContProj
from src.loss.loss import ImageLoss,GCCLoss,PoseLoss


import wandb


def train(recon_net,pose_net,device,config,trainloader,valloader):

    wandb.init(project='v1',reinit=True)
    print("training model!")
    # loss_criterion = torch.nn.MSELoss()

    # loss_criterion.to(device)

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
            'lr': config['learning_rate_recon_net']
        },
        {
            # TODO: optimizer params and learning rate for latent code (lr provided in config)
            'params': pose_net.parameters(),
            'lr': config['learning_rate_pose_net']
        }
    ])

    ## Setting GPU
    recon_net.to(device)
    pose_net.to(device)

    recon_net.train()
    pose_net.train()

    best_accuracy = 0.

    train_loss_running = 0.
    num_epochs = config['max_epochs']
    if(config['use_pretrained']):
      print('Using Pre Trained Model!')
      num_epochs = 200

    for epoch in range(num_epochs):
        train_loss_running = [] 
        print('**********************************************')
        print('Epoch : ', epoch)
        for i, batch in enumerate(trainloader):
            
            
            # move batch to device
            ShapeNet.move_batch_to_device(batch, device)
            optimizer.zero_grad()

            batch['img_mask'] = 1 - torchvision.transforms.Normalize(batch['img_mask'][0].mean(), batch['img_mask'][0].std())(batch['img_mask'][0])
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
                # pcl_out_rot = world2cam(pcl_xyz, pose_out[:, 0], pose_out[:, 1], 2., 2., 1,device="cuda")
                # pcl_out_pers = perspective_transform(pcl_out_rot, 1, device="cuda")
                # img_out = get_proj_rgb(pcl_out_pers, pcl, 1024, 60, 60, device="cuda")
                # mask_out = get_proj_mask(pcl_out_pers, 60, 60, 1024, 0.4, device="cuda")
                # print(idx)
                # pcl_out_rot.append(world2cam(pcl_out[0], pose_all[:, idx, 0],
                #     pose_all[:,idx,1], 2., 2., len(batch),config["device"]))
                # pcl_out_persp.append(perspective_transform(pcl_out_rot[idx],
                #     len(batch),config["device"]))
                # img_out.append(get_proj_rgb(pcl_out_persp[idx], pcl_rgb_out[0], 1024,
                #     224, 224,1., 100, 'rgb',config["device"])[0])
                # mask_out.append(get_proj_mask(pcl_out_persp[idx], 224, 224,
                #     1024, 0.4,config["device"]))

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
            # print("UM",img_out[0][0].shape)

            # temp_img = torch.clone(img_out[0][0])
            # temp_mask = torch.clone(mask_out[0])
            # images = wandb.Image((temp_img.cpu().detach().numpy()*255).astype(np.uint8), caption="Projected Image")
            # masks = wandb.Image((temp_mask.cpu().detach().numpy()*255).astype(np.uint8), caption="Projected Mask")
            # log_list = [images, masks]
            # wandb.log({"image": log_list})
            temp_img = torch.clone(img_out[0][0])
            temp_mask = torch.clone(mask_out[0])
            temp_gray_img = torchvision.transforms.Grayscale()(torch.permute(temp_img,[2,0,1]))
            temp_gray_img = torch.permute(temp_gray_img,[1,2,0])
            
            # print(torch.squeeze(torch.clone(pcl_out[0]),0).T.shape)
            temp_pcl_xyz = torch.squeeze(torch.clone(pcl_out[0]),0).T.cpu().detach().numpy()
            temp_pcl_rgb = torch.squeeze(torch.clone(pcl_rgb_out[0]),0).T.cpu().detach().numpy()
            
    #         point_cloud = wandb.Object3D({'type':'lidar/beta',
    #                                      'points':temp_pcl})
            images = wandb.Image((temp_gray_img.cpu().detach().numpy()*255).astype(np.uint8),
                                caption="Projected Image")
            masks = wandb.Image((temp_mask.cpu().detach().numpy()*255).astype(np.uint8), caption="Projected Mask")
            log_list = [images,masks]
            wandb.log({"image": log_list})
            
            wandb.log({"point_cloud_1" : [wandb.Object3D(temp_pcl_xyz),wandb.Object3D(temp_pcl_rgb)]})


            # print(img_out[1][0].shape)
            # Reconstruct the point cloud from and predict the pose of projected images
            for idx in range(config['n_proj']):
                # print(idx)
                # print('Pojected images : ', img_out[idx].shape)
                # print(torch.permute(img_out[idx][0],[0, 3, 1, 2]).shape)
                # temp_img = torch.permute(img_out[0][idx],[0, 3, 1, 2]).contiguous()
                pcl_xyz, pcl_rgb = recon_net(torch.permute(img_out[idx],[0, 3, 1, 2]).contiguous())
                pcl_out.append(pcl_xyz)
                pcl_rgb_out.append(pcl_rgb)

                pose_out.append(pose_net(torch.permute(img_out[idx],[0, 3, 1, 2]).contiguous()))

            temp_pcl = torch.squeeze(torch.clone(pcl_out[1]),0).T.cpu().detach().numpy()
#         temp_pcl2 = torch.squeeze(torch.clone(pcl_out[1]),0).T.cpu().detach().numpy()
            gt_pcl = torch.squeeze(torch.clone(batch['pcl']),0).T.cpu().detach().numpy()
            wandb.log({"point_cloud_2" : [wandb.Object3D(temp_pcl),wandb.Object3D(gt_pcl)]})

            # Define Losses
            # 2D Consistency Loss - L2
            # print('IMAGE LOSS : ',torch.permute(torch.stack(img_out)[0],[0, 3, 1, 2]).contiguous().shape)
            img_ae_loss, _, _ = img_loss(batch['img_rgb'], torch.permute(torch.stack(img_out)[0],[0, 3, 1, 2]).contiguous()
                        , 'l2_sq')
            # print('MASK CHECK : ',torch.squeeze(batch['img_mask'],1).shape,torch.stack(mask_out)[0].shape)
            mask_ae_loss, mask_fwd, mask_bwd = img_loss(torch.squeeze(batch['img_mask'],1), torch.stack(mask_out)[0],
                    'bce', affinity_loss=True)
            # print('Mask Loss Check ')

            # Pose Loss
            pose_loss_pose = pose_loss(pose_ip, torch.stack(pose_out[2:], axis=1), 'l1')

            # 3D Consistency Loss
            consist_3d_loss = 0.
            for idx in range(config['n_proj']):
                # if args._3d_loss_type == 'adj_model':
                #     consist_3d_loss += get_3d_loss(pcl_out[idx], pcl_out[idx+1], 'chamfer')
                # elif args._3d_loss_type == 'init_model':
                consist_3d_loss += chamfer_distance(pcl_out[idx], pcl_out[0])[0]

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
            symm_loss = chamfer_distance(pcl_pos, pcl_neg)[0]

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

            
            # loss.backward(retain_graph=True)
            # recon_loss.backward(retain_graph=True)
            # pose_loss_val.backward(retain_graph=True)
            total_loss.backward(retain_graph=True)

            optimizer.step()

            print('Iteration : ',i)
            print('Loss : ',total_loss.item())
            wandb.log({'Total Loss': total_loss.item()})
            wandb.log({'Recon_loss': recon_loss.item()})
            wandb.log({'Pose_loss': pose_loss_val.item()})
            # print('recon_loss : ',recon_loss.item())
            # print('pose_loss : ',pose_loss_val.item())

            train_loss_running.append(total_loss.item())

            ## VALIDATION
        if(not config['use_pretrained']):
          if(epoch ==0 or (sum(train_loss_running)/ len(train_loss_running))<best_loss):
            print('Saving new model!')
            best_loss = sum(train_loss_running)/ len(train_loss_running)
            torch.save(recon_net.state_dict(), f'src/runs/{config["experiment_name"]}/recon_model_best.ckpt')
            torch.save(pose_net.state_dict(), f'src/runs/{config["experiment_name"]}/pose_model_best.ckpt')

            # Saving to GDrive
            torch.save(recon_net.state_dict(), f'../drive/MyDrive/recon_model_best.ckpt')
            torch.save(pose_net.state_dict(), f'../drive/MyDrive/pose_model_best.ckpt')


            



def main(config):

    trainset = ShapeNet('train' if not config['is_overfit'] else 'overfit', config['category'], config['n_proj'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)

    valset = ShapeNet('val' if not config['is_overfit'] else 'overfit', config['category'], config['n_proj'])
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


    train(recon_net,pose_net,device,config,trainloader,valloader)