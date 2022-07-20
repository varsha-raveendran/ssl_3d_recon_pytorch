from pathlib import Path

import numpy as np
import torch

from src.data.shapenet import ShapeNet
from src.network_architecture.recon_model import ReconstructionNet
from pytorch3d.loss import chamfer_distance
import open3d as o3d
import wandb


def train(config):
    # create dataloaders
    wandb.init(project=config['experiment_name'],reinit=True)

    trainset = ShapeNet('train' if not config['is_overfit'] else 'overfit', config['category'], config['n_proj'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    valset = ShapeNet('val' if not config['is_overfit'] else 'overfit', config['category'], config['n_proj'])
    valloader = torch.utils.data.DataLoader(valset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    device = config['device']
    # declare device
    # device = torch.device('cpu')
    device = torch.device(config['device'])    
    model = ReconstructionNet()
    
    model.to(device)
    wandb.watch(model)

    loss_criterion = torch.nn.MSELoss()


    loss_criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'] )

    model.train()

    best_accuracy = 0.

    train_loss_running = 0.
    prediction = None
    name = None

    for epoch in range(config['max_epochs']):
        train_loss_running = 0 
        for i, batch in enumerate(trainloader):
            # move batch to device
            ShapeNet.move_batch_to_device(batch, device)
            optimizer.zero_grad()
            pred_img, pred_mask = model(batch['img_rgb'])
            print("prediction shape: ", pred_img.shape)
            #loss_total = loss_criterion(pred_img, batch['pcl'])
            loss_total , _ = chamfer_distance(pred_img, batch['pcl'])
            loss_total.backward()

            optimizer.step()

            # loss logging
            train_loss_running += loss_total.item()
            iteration = epoch * len(trainloader) + i
            print("Loss: " , train_loss_running)
            
            """if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                train_loss_running = 0."""

            # validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):

                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                model.eval()
                
                loss_total_val = 0
                # forward pass and evaluation for entire validation set
                for batch_val in valloader:
                    ShapeNet.move_batch_to_device(batch_val, device)

                    with torch.no_grad():
                        print(batch_val['img_rgb'].shape)
                        pred_img, pred_mask = model(batch_val['img_rgb'])

                    #loss_total_val = loss_criterion(pred_img, batch_val['pcl']).item()
                    loss_total_val, _ = chamfer_distance(pred_img, batch_val['pcl'])

                # add metric
                wandb.log({"train_loss": train_loss_running})
                
                wandb.log({"val_loss": loss_total_val})
                
                # temp_pcl_xyz = torch.permute(torch.squeeze(torch.clone(pred_mask),0).T, [2,0,1])[0].cpu().detach().numpy()
                # temp_pcl_rgb = torch.permute(torch.squeeze(torch.clone(pred_img),0).T, [2,0,1])[0].cpu().detach().numpy()
                print(pred_img.shape)
                #temp_pcl_xyz = torch.permute(torch.squeeze(torch.clone(pred_mask),0).T, [0,1]).cpu().detach().numpy()
                #temp_pcl_rgb = torch.permute(torch.squeeze(torch.clone(pred_img),0).T, [0,1]).cpu().detach().numpy()
                temp_pcl_rgb = torch.reshape(torch.clone(torch.squeeze(pred_img[0])), ( 1024, 3)).cpu().detach().numpy()
                temp_pcl_xyz =  torch.reshape(torch.clone(torch.squeeze(pred_mask[0])), ( 1024, 3)).cpu().detach().numpy()
                images = wandb.Image((batch_val['img_rgb'][0][0].cpu().detach().numpy()*255).astype(np.uint8), caption="Ground Truth image")
                # temp_pcl_xyz1 = torch.permute(torch.squeeze(torch.clone(pred_mask),0).T, [2,0,1])[0].cpu().detach().numpy()
                # temp_pcl_rgb1 = torch.permute(torch.squeeze(torch.clone(pred_img),0).T, [2,0,1])[0].cpu().detach().numpy()
                # wandb.log({"point_cloud_1" : [wandb.Object3D(temp_pcl_xyz1),
                # wandb.Object3D(temp_pcl_rgb1)]})
                # wandb.log({"point_cloud_2" : [wandb.Object3D(temp_pcl_xyz),wandb.Object3D(temp_pcl_rgb)]})
                log_list = [images]
                wandb.log({"image": log_list})
                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_total_val / len(valloader):.3f}')
                prediction = model(torch.unsqueeze(batch['img_rgb'][0], 0))
                wandb.log({"point_cloud_pred" : [wandb.Object3D(torch.reshape(prediction[0], ( 1024, 3)).cpu().detach().numpy())]})
                name = batch['name'][0]
                print(prediction[0].shape)
                # set model back to train
                model.train()
    wandb.finish()
    torch.save(model.state_dict(), f'src/runs/{config["experiment_name"]}/model_best.ckpt')
    pcd = o3d.geometry.PointCloud()

    data_pcl = torch.reshape(prediction[0], ( 1024, 3)).cpu().detach().numpy()
    print(data_pcl.shape)
    print(name)

    pcd.points = o3d.utility.Vector3dVector(data_pcl)
    o3d.io.write_point_cloud("./data_3.ply", pcd)


