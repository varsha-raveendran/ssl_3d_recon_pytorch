from pathlib import Path
import json
from os.path import join, abspath, basename
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms




class ShapeNet(torch.utils.data.Dataset):
    
    rendered_path = Path("src/data/ShapeNet_rendered")  
    pcl_path = Path("src/data/ShapeNet_v1")
    
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
        rgb_image = Image.open(self.rendered_path / Path(self.category)/ filename / Path('render_%s.png'%render_id))
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        img_tensor = transform(rgb_image)
        print(img_tensor.shape)
        
        mask_image = Image.open(self.rendered_path / Path(self.category) / filename / Path('depth_%s.png'%render_id))
        mask_tensor = transform(mask_image)
        
        #Load point cloud
        pcl = np.load(self.pcl_path / Path(self.category) / filename / 'pointcloud_1024.npy')

        return {
            'name': filename,
            'img_rgb' : img_tensor,
            'img_mask' : mask_tensor,
            'pcl' : pcl,
            #'pose' : 
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        return batch.to(device)

    
