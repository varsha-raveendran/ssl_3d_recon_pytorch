from pathlib import Path

import numpy as np
import torch

from src.data.shapenet import ShapeNet
from src.network_architecture.recon_model import ReconstructionNet

import wandb


def train(config):
    # create dataloaders
    wandb.init(project='recon_run',reinit=True)

    trainset = ShapeNet('train' if not config['is_overfit'] else 'overfit', config['category'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    valset = ShapeNet('val' if not config['is_overfit'] else 'overfit', config['category'])
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

    for epoch in range(config['max_epochs']):
        train_loss_running = 0 
        for i, batch in enumerate(trainloader):
            # move batch to device
            ShapeNet.move_batch_to_device(batch, device)
            optimizer.zero_grad()
            prediction = model(batch['img_rgb'])
            print("prediction shape: ", prediction.shape)
            loss_total = loss_criterion(prediction, batch['pcl'])
            loss_total.backward()

            optimizer.step()

            # loss logging
            train_loss_running += loss_total.item()
            iteration = epoch * len(trainloader) + i
            
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
                        prediction = model(batch_val['img_rgb'])

                    loss_total_val = loss_criterion(prediction, batch_val['pcl']).item()

                # add metric
                wandb.log({"train_loss": train_loss_running})
                
                wandb.log({"val_loss": loss_total_val})
                
                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_total_val / len(valloader):.3f}')
                
                # set model back to train
                model.train()
    wandb.finish()
    torch.save(model.state_dict(), f'src/runs/{config["experiment_name"]}/model_best.ckpt')

