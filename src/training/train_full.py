from turtle import pos
import numpy as np
import torch
from pathlib import Path

from src.data.shapenet import ShapeNet
from src.network_architecture.recon_model import ReconstructionNet
from src.network_architecture.pose_net import PoseNet
from src.renderer.projection import World2Cam, PerspectiveTransform, RgbContProj, ContProj
from src.loss.loss import ImageLoss,GCCLoss,PoseLoss


import wandb


def train(recon_net,pose_net,device,config,trainloader,valloader):


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

    for epoch in range(config['max_epochs']):
        train_loss_running = 0 
        for i, batch in enumerate(trainloader):
            # move batch to device
            ShapeNet.move_batch_to_device(batch, device)
            optimizer.zero_grad()


            pcl_out = pose_out = img_out = pcl_rgb_out = []
            pcl_out_rot = pcl_out_persp = mask_out = []

            pcl_out.append(pcl_xyz)
            pcl_rgb_out.append(pcl_rgb)

            ## GET RECONSTRUCTION FROM INPUT IMAGE
            pcl_xyz,pcl_rgb = recon_net(batch['img_rgb'])

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
                pcl_out_rot.append(world2cam(pcl_out[0], pose_all[:, idx, 0],
                    pose_all[:,idx,1], 2., 2., len(batch),device="cuda"))
                pcl_out_persp.append(perspective_transform(pcl_out_rot[idx],
                    len(batch),device="cuda"))
                img_out.append(get_proj_rgb(pcl_out_persp[idx], pcl_rgb_out[0], 1024,
                    64, 64,device="cuda")[0])
                mask_out.append(get_proj_mask(pcl_out_persp[idx], 64, 64,
                    1024, 0.4,device="cuda"))
            
            # Reconstruct the point cloud from and predict the pose of projected images
            for idx in range(config['n_proj']):
                pcl_xyz, pcl_rgb = recon_net(img_out[idx])
                pcl_out.append(pcl_xyz)
                pcl_rgb_out.append(pcl_rgb)

                pose_out.append(pose_net(img_out[idx]))

            # Define Losses
            # 2D Consistency Loss - L2
            img_ae_loss, _, _ = img_loss(batch['img_rgb'], img_out[0], 'l2_sq')
            mask_ae_loss, mask_fwd, mask_bwd = img_loss(batch['img_mask'], mask_out[0],
                    'bce', affinity_loss=True)

            # 3D Consistency Loss
            consist_3d_loss = 0.
            for idx in range(config['n_proj']):
                # if args._3d_loss_type == 'adj_model':
                #     consist_3d_loss += get_3d_loss(pcl_out[idx], pcl_out[idx+1], 'chamfer')
                # elif args._3d_loss_type == 'init_model':
                consist_3d_loss += gcc_loss(pcl_out[idx], pcl_out[0], 'chamfer')

            # Pose Loss
            pose_loss_pose = pose_loss(pose_ip, torch.stack(pose_out[2:], axis=1), 'l1')

            ##TODO
            # Symmetry loss - assumes symmetry of point cloud about z-axis
            # # Helps obtaining output aligned along z-axis
            # pcl_y_pos = (pcl_out[0][:,:,1:2]>0).type(torch.FloatTensor)
            # pcl_y_neg = tf.to_float(pcl_out[0][:,:,1:2]<0)
            # pcl_pos = pcl_y_pos*tf.concat([pcl_out[0][:,:,:1], tf.abs(pcl_out[0][:,:,1:2]),
            #         pcl_out[0][:,:,2:3]], -1)
            # pcl_neg = pcl_y_neg*tf.concat([pcl_out[0][:,:,:1], tf.abs(pcl_out[0][:,:,1:2]),
            #         pcl_out[0][:,:,2:3]], -1)
            # symm_loss = get_chamfer_dist(pcl_pos, pcl_neg)[-1]

            # Total Loss
            loss = (config['lambda_ae']*img_ae_loss) + (config['lambda_3d']*consist_3d_loss) +\
                    (config['lambda_pose']*pose_loss_pose)
            recon_loss = (config['lambda_ae']*img_ae_loss) + (config['lambda_3d']*consist_3d_loss)\
                            + (config['lambda_ae_mask']*mask_ae_loss) +\
                            (config['lambda_mask_fwd']*mask_fwd) + (config['lambda_mask_bwd']*mask_bwd)
            # if args.symmetry_loss:
            #     recon_loss += (config['lambda_symm']*symm_loss)
            pose_loss = (config['lambda_ae_pose']*img_ae_loss) + (config['lambda_pose']*pose_loss_pose)\
                            + (config['lambda_mask_pose']*mask_ae_loss)

            
            loss.backward()
            recon_loss.backward()
            pose_loss.backward()

            optimizer.step()

            print('Iteration : ',i)
            print('Loss : ',loss.item())
            print('recon_loss : ',recon_loss.item())
            print('pose_loss : ',pose_loss.item())

            ## VALIDATION


            



def main(config):

    trainset = ShapeNet('train' if not config['is_overfit'] else 'overfit', config['category'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    valset = ShapeNet('val' if not config['is_overfit'] else 'overfit', config['category'])
    valloader = torch.utils.data.DataLoader(valset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # device = 
    # declare device
    device = torch.device(config['device'])  ## CHANGE TO CUDA  
    recon_net = ReconstructionNet()
    pose_net = PoseNet() ## Need to merge
    pose_net = None

    model = ReconstructionNet() 
    
    # model.to(device)

    train(recon_net,pose_net,device,config,trainloader,valloader)