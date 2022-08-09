# ssl_3d_recon_pytorch

Pytorch Implementation of From Image Collections to Point Clouds with Self-supervised Shape and Pose Networks. 

## Installing Pytorch3D
We recommend using Anaconda to create a new environment to install Pytorch3d.
```
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d-nightly
pip install wandb pandas opencv-python pillow
```

## Dataset
We use the <a href="https://github.com/shubhtuls/drc/blob/master/docs/snet.md#rendering" target="_blank" >code</a> provided by Tulsiani et al. to obtain the rendered images and part segmentation maps. 
Download links for the ShapeNet point cloud dataset is provided below: <br>
ShapeNet pointclouds (~2.8 GB): https://drive.google.com/open?id=1cfoe521iTgcB_7-g_98GYAqO553W8Y0g <br>

## Training
Modify the config dictionary inside run.py and execute the following: 
```
python run.py
```

## Evaluation
First save the point clouds 
```
python run_save.py
```
Then change directory to src/evaluation/ and run
```
python get_metric_pcl.py
```




@inproceedings{navaneet2020ssl3drecon,
 author = {Navaneet, K L and Mathew, Ansu and Kashyap, Shashank and Hung, Wei-Chih and Jampani, Varun and Babu, R Venkatesh},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 title = {From Image Collections to Point Clouds with Self-supervised Shape and Pose Networks},
 year = {2020}
}
