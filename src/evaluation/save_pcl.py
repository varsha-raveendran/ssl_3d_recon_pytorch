import numpy as np
import pandas as pd
import torch
import torchvision
import os
from pytorch3d.loss import chamfer_distance

from src.data.shapenet import ShapeNet
from src.network_architecture.recon_model import ReconstructionNet
from src.network_architecture.pose_net import PoseNet
from src.renderer.projection import World2Cam, PerspectiveTransform, RgbContProj, ContProj
from src.loss.loss import ImageLoss,GCCLoss,PoseLoss

def optimise_and_save(item, config):
    device = torch.device(config['device'])
    #Get the item variables we need in the correct dimensions
    img_rgb = torch.unsqueeze(item['img_rgb'], 0).to(device)
    img_mask = torch.unsqueeze(item['img_mask'], 0).to(device)
    img_mask = 1 - torchvision.transforms.Normalize(img_mask[0].mean(), img_mask[0].std())(img_mask[0])
    # random_pose = torch.unsqueeze(item['random_pose'], 0).to(device)
    gt_pose = torch.unsqueeze(torch.from_numpy(item['gt_pose']), 0).to(device)

    #Initialise models and load trained checkpoints
    recon_net = ReconstructionNet()
    pose_net = PoseNet()
    ckpt_recon = f'src/runs/{config["experiment_name"]}/recon_model_best.ckpt'
    recon_net.load_state_dict(torch.load(ckpt_recon, map_location='cpu'))
    ckpt_pose = f'src/runs/{config["experiment_name"]}/pose_model_best.ckpt'
    pose_net.load_state_dict(torch.load(ckpt_pose, map_location='cpu'))

    #Create initial prediction for ISO loss
    recon_net.to(device)
    recon_net.eval()

    initial_pcl = recon_net(img_rgb.to(device))
    #Detach point cloud from compute graph
    initial_pcl = initial_pcl[0].detach().cpu().numpy()

    #Re-Convert point cloud to tensor and move to device
    initial_pcl = torch.from_numpy(initial_pcl).to(device)

    #Initialize projection modules
    world2cam = World2Cam()
    perspective_transform = PerspectiveTransform()
    get_proj_rgb = RgbContProj()
    get_proj_mask = ContProj()

    #Define Losses
    img_loss = ImageLoss()
    gcc_loss = GCCLoss()
    pose_loss = PoseLoss()

    #Initialise optimiser
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

    #Setting GPU
    recon_net.to(device)
    pose_net.to(device)

    recon_net.train()
    pose_net.train()

    best_accuracy = 0.

    train_loss_running = 0.
    num_epochs = config['max_epochs']
    
    print("Optimising item: ", item['name'])

    ## Logging metrics for training epochs
    log_total_loss = []
    log_pose_loss = []
    log_recon_loss = []
    log_symm_loss = []

    for epoch in range(num_epochs):
        train_loss_running = []
        pose_loss_running = []
        recon_loss_running = [] 
        symm_loss_running = []
        print('**********************************************')
        print('Step : ', epoch)
            
        optimizer.zero_grad()

        # img_mask = 1 - torchvision.transforms.Normalize(img_mask[0].mean(), img_mask[0].std())(img_mask[0])

        pcl_out_rot = []
        pcl_out_persp = []
        mask_out = []
        pcl_out = []
        pose_out = []
        img_out = []
        pcl_rgb_out = []

        ## GET RECONSTRUCTION FROM INPUT IMAGE
        pcl_xyz,pcl_rgb = recon_net(img_rgb)
        pcl_out.append(pcl_xyz)
        pcl_rgb_out.append(pcl_rgb)

        pose_ip = torch.unsqueeze(item['random_pose'], 0).to(device) ## Batch Size x 2 x 2
        ## USE POSENET TO GET POSE PREDICTIONS (B,2)
        pose_out.append(pose_net(img_rgb))
        pose_all = torch.concat([torch.unsqueeze(pose_out[0], axis=1), pose_ip], axis=1)


        ## USE THE PROJECTION MODULE TO GET PROJECTIONS FROM PC1 AND POSES
        for idx in range(config['n_proj']):
            pcl_out_rot = world2cam(pcl_out[0], pose_all[:, idx, 0],
                pose_all[:,idx,1], 2., 2., img_rgb.shape[0],config["device"])
            pcl_out_persp = perspective_transform(pcl_out_rot,
                img_rgb.shape[0],config["device"])    
            temp_img_out = get_proj_rgb(pcl_out_persp, pcl_rgb_out[0], 1024,
                64, 64,1., 100, 'rgb',config["device"])
            img_out.append(temp_img_out[0])
            mask_out.append(get_proj_mask(pcl_out_persp, 64, 64,
                1024, 0.4,config["device"])) ### Invert mask and image(?)

        # Reconstruct the point cloud from and predict the pose of projected images
        for idx in range(config['n_proj']):
            pcl_xyz, pcl_rgb = recon_net(torch.permute(img_out[idx],[0, 3, 1, 2]).contiguous())
            pcl_out.append(pcl_xyz)
            pcl_rgb_out.append(pcl_rgb)

            pose_out.append(pose_net(torch.permute(img_out[idx],[0, 3, 1, 2]).contiguous()))

        # Define Losses
        # 2D Consistency Loss - L2
        img_ae_loss, _, _ = img_loss(img_rgb, torch.permute(torch.stack(img_out)[0],[0, 3, 1, 2]).contiguous()
                    , 'l2_sq')
        # print('MASK CHECK : ',torch.squeeze(batch['img_mask'],1).shape,torch.stack(mask_out)[0].shape)
        mask_ae_loss, mask_fwd, mask_bwd = img_loss(torch.squeeze(img_mask,1), torch.stack(mask_out)[0],
                'bce', affinity_loss=True)

        # Pose Loss
        # pose_all = torch.permute(pose_all,[1,0,2])
        # pose_loss_pose = 0
        # for idx in range(config['n_proj']):
        #     pose_loss_pose += pose_loss(pose_all[idx], torch.stack(pose_out)[idx+1], 'l1')
        # pose_loss_pose /= config['n_proj']

        # 3D Consistency Loss
        consist_3d_loss = 0.
        for idx in range(config['n_proj']):
            consist_3d_loss += chamfer_distance(torch.permute(pcl_out[idx],[0,2,1]),torch.permute(pcl_out[0],[0,2,1]))[0]

        # Symmetry loss - assumes symmetry of point cloud about z-axis
        # # Helps obtaining output aligned along z-axis
        pcl_y_pos = (pcl_out[0][:,:,1:2]>0).type(torch.FloatTensor).to(device)
        pcl_y_neg = (pcl_out[0][:,:,1:2]<0).type(torch.FloatTensor).to(device)
        pcl_pos = pcl_y_pos*torch.concat([pcl_out[0][:,:,:1], torch.abs(pcl_out[0][:,:,1:2]),
                pcl_out[0][:,:,2:3]], -1)
        pcl_neg = pcl_y_neg*torch.concat([pcl_out[0][:,:,:1], torch.abs(pcl_out[0][:,:,1:2]),
                pcl_out[0][:,:,2:3]], -1)
        symm_loss = chamfer_distance(torch.permute(pcl_pos,[0,2,1]), torch.permute(pcl_neg,[0,2,1]))[0]
        # symm_loss = 0.
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
 
        for idx in range(config['n_proj']):
          reg_loss += chamfer_distance(torch.permute(initial_pcl,[0,2,1]), torch.permute(pcl_out[idx],[0,2,1]))[0]
                
        total_loss = (config['lambda_ae']*img_ae_loss) + (config['lambda_3d']*reg_loss)\
                        + (config['lambda_ae_mask']*mask_ae_loss) +\
                        (config['lambda_mask_fwd']*mask_fwd) + (config['lambda_mask_bwd']*mask_bwd) +\
                                (config['lambda_symm']*symm_loss)  

        # loss.backward(retain_graph=True)
        # recon_loss.backward(retain_graph=True)
        # pose_loss_val.backward(retain_graph=True)
        total_loss.backward(retain_graph=True)

        optimizer.step()

        print('Iteration : ',epoch)
        print('Loss : ',total_loss.item())
        # print('recon_loss : ',recon_loss.item())
        # print('pose_loss : ',pose_loss_val.item())

        train_loss_running.append(total_loss.item())
        recon_loss_running.append(recon_loss.item())
        # if config["use_symmetry_loss"]:
        #     symm_loss_running.append(symm_loss.item())

        # Uncomment if we wanna do best model
        # if(epoch ==0 or (sum(train_loss_running)/ len(train_loss_running))<best_loss):
        #     print('Saving new model!')
        #     best_loss = sum(train_loss_running)/ len(train_loss_running)
        #     torch.save(recon_net.state_dict(), f'src/runs/{config["experiment_name"]}/recon_model_inf.ckpt')
        #     torch.save(pose_net.state_dict(), f'src/runs/{config["experiment_name"]}/pose_model_inf.ckpt')

        #     # Saving to GDrive
        #     torch.save(recon_net.state_dict(), f'../drive/MyDrive/recon_model_inf.ckpt')
        #     torch.save(pose_net.state_dict(), f'../drive/MyDrive/pose_model_inf.ckpt')

    # Now save       
    if not os.path.exists('output'):
        os.makedirs('output')

    recon_net.eval()
    output_pcl, _= recon_net(img_rgb)
    export_pointcloud_to_obj(f'output/{item["name"]}.obj', torch.squeeze(output_pcl,0).T.cpu().detach().numpy())    


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """
    file = open(path, 'w')
    for v in pointcloud:
        val = 'v ' + str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n'
        file.write(val)
    file.close()


def main(config):
    trainset = ShapeNet('test', config['category'], config['n_proj'])

    for item in trainset:
        optimise_and_save(item, config)