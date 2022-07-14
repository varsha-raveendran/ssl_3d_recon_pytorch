from turtle import pos
import numpy as np
import torch
from pathlib import Path

from src.data.shapenet import ShapeNet
from src.network_architecture.recon_model import ReconstructionNet
from src.renderer.projection import World2Cam, PerspectiveTransform, RgbContProj, ContProj
from src.loss.loss import ImageLoss,GCCLoss,PoseLoss


import wandb


def train(recon_net,pose_net,device,config,trainloader,valloader):


    print("training model!")
    # loss_criterion = torch.nn.MSELoss()

    # loss_criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'] )

    model.train()

    best_accuracy = 0.

    train_loss_running = 0.

    for epoch in range(config['max_epochs']):
        train_loss_running = 0 
        for i, batch in enumerate(trainloader):
            # move batch to device
            ShapeNet.move_batch_to_device(batch, device)
            optimizer.zero_grad()


            ## GET RECONSTRUCTION FROM INPUT IMAGE

            recon_prediction1 = model(batch['img_rgb'])

            ## USE POSENET TO GET POSE PREDICTIONS (B,2)

            ## USE THE PORJECTION MODULE TO GET PROJECTIONS FROM PC1 AND MASKS

            # Initialize projection modules
            world2cam = World2Cam()
            perspective_transform = PerspectiveTransform()
            get_proj_rgb = RgbContProj()
            get_proj_mask = ContProj()



            ## USE THE PROJECTIONS AS INPUT TO RECONNET AND POSENET

            ## v0,1 : rnadomly 


            ## GET LOSSES

            ## LOSS BACKWARD

            ## OPTIM STEP

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
    # pose_net = PoseNet() ## Need to merge
    pose_net = None

    # model = ReconstructionNet() 
    
    # model.to(device)

    train(recon_net,pose_net,device,config,trainloader,valloader)