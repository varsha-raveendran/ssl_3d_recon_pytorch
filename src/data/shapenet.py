from pathlib import Path
import json
from os.path import join, abspath, basename
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2

class ShapeNet(torch.utils.data.Dataset):
    
    rendered_path = "src/data/ShapeNet_rendered"  
    pcl_path = "src/data/ShapeNet_v1"
    
    with open("src/data/shape_info.json") as json_file:
        class_name_mapping = json.load(json_file)
        print(class_name_mapping)

    def __init__(self, split, category):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.category = category
        self.items = np.load('src/data/splits/shapenet/images_list_%s_%s.npy'%(category, split), allow_pickle=True)
        print(self.items.shape)
        self.image_ids = [item[0].split('_')[1] for item in self.items]
        self.file_names = [item[0].split('_')[0] for item in self.items]
        
    def __getitem__(self, index):
        
        filename = self.file_names[index]
        render_id = self.image_ids[index]
        print(render_id)
        
        #Load RGB view
        # rgb_image = Image.open(self.rendered_path / Path(self.category)/ filename / Path('render_%s.png'%render_id))
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        img_path = join(self.rendered_path , f'{self.category}/{filename}/render_{render_id}.png')
        rgb_image =  cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        img_tensor = transform(rgb_image)
        print(img_tensor.shape)
        
        mask_image = Image.open(self.rendered_path / Path(self.category) / filename / Path('depth_%s.png'%render_id))
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        #mask_path = join(self.rendered_path, f'{self.category}/{filename}/depth_{render_id}.png')
        #mask_image =  cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask_tensor = transform(mask_image)
        
        
        
        #Load point cloud
        pcl = np.load(self.pcl_path / Path(self.category) / filename / 'pointcloud_1024.npy').astype(np.float32)
        pcl = torch.from_numpy(pcl)
        
        return {
            'name': filename,
            'img_rgb' : img_tensor.float() ,
            'img_mask' : mask_tensor.float() ,
            'pcl' : torch.reshape(pcl, ( 3, 1024)),
            #'pose' : 
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['img_rgb'] = batch['img_rgb'].to(device)
        batch['img_mask'] = batch['img_mask'].to(device)
        batch['pcl'] = batch['pcl'].to(device)

    
