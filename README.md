# Multimodal Virtual Point 3D Detection

Turning pixels into virtual points for multimodal 3D object detection. 

<p align="center"> <img src='docs/teaser.png' align="center" height="230px"> </p>

> [**Multimodal Virtual Point 3D Detection**](https://tianweiy.github.io/mvp/),            
> Tianwei Yin, Xingyi Zhou, Philipp Kr&auml;henb&uuml;hl,        
> *arXiv technical report ([arXiv 2111.06881 ](https://arxiv.org/abs/2111.06881))*  



    @article{yin2021multimodal,
      title={Multimodal Virtual Point 3D Detection},
      author={Yin, Tianwei and Zhou, Xingyi and Kr{\"a}henb{\"u}hl, Philipp},
      journal={NeurIPS},
      year={2021},
    }

## Contact
Any questions or suggestions are welcome! 

Tianwei Yin [yintianwei@utexas.edu](mailto:yintianwei@utexas.edu) 
Xingyi Zhou [zhouxy@cs.utexas.edu](mailto:zhouxy@cs.utexas.edu)

## Abstract
Lidar-based sensing drives current autonomous vehicles. Despite rapid progress, current Lidar sensors still lag two decades behind traditional color cameras in terms of resolution and cost. For autonomous driving, this means that large objects close to the sensors are easily visible, but far-away or small objects comprise only one measurement or two. This is an issue, especially when these objects turn out to be driving hazards. On the other hand, these same objects are clearly visible in onboard RGB sensors. In this work, we present an approach to seamlessly fuse RGB sensors into Lidar-based 3D recognition. Our approach takes a set of 2D detections to generate dense 3D virtual points to augment an otherwise sparse 3D point-cloud. These virtual points naturally integrate into any standard Lidar-based 3D detectors along with regular Lidar measurements. The resulting multi-modal detector is simple and effective. Experimental results on the large-scale nuScenes dataset show that our framework improves a strong CenterPoint baseline by a significant 6.6 mAP, and outperforms competing fusion approaches.

## Main results

#### 3D detection on nuScenes validation set

|         |  MAP ↑  | NDS ↑  |
|---------|---------|--------|
|CenterPoint-Voxel |  59.5   | 66.7 |    
|CenterPoint-Voxel + MVP | **66.0** | **69.9** | 
|CenterPoint-Pillar |  52.4   | 61.5 |    
|CenterPoint-Voxel + MVP | **62.8** | **66.2** | 

#### 3D detection on nuScenes test set 

|         |  MAP ↑  | NDS ↑  | PKL ↓  |
|---------|---------|--------|--------|
|MVP |  66.4   | 70.5   | 0.603   |    

## Use MVP 

### Installation

Please install [CenterPoint](https://github.com/tianweiy/CenterPoint/blob/master/docs/INSTALL.md) and [CenterNet2](https://github.com/xingyizhou/CenterNet2). 
Make sure to add a link to [CenterNet2](https://github.com/xingyizhou/CenterNet2/tree/master/projects/CenterNet2) in your python path.
We will use CenterNet2 for 2D instance segmentation and CenterPoint for 3D detection. 

### Getting Started

#### Download nuscenes data and organise as follows

```
# For nuScenes Dataset         
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata
```

Create a symlink to the dataset root in both CenterPoint and MVP's root directories. 
```bash
mkdir data && cd data
ln -s DATA_ROOT nuScenes
```
Remember to change the DATA_ROOT to the actual path in your system. 

#### Generate Virtual Points 

Download the centernet2 model from [here](https://drive.google.com/file/d/1k-uPZJq5mVl9Y5z88fyurxxIoLmuVfZ7/view?usp=sharing) and place it in the root directory.

Use the following command in the current directory to generate virtual points for nuscenes training and validation sets. The points will be saved to ```data/nuScenes/samples or sweeps/LIDAR_TOP_VIRTUAL```. 

```bash
python virtual_gen.py --info_path data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl  
```

You will need about 80GB space and the whole process will take 10 to 20 hours using a single GPU. You can also download the precomputed virtual points from [here](https://drive.google.com/file/d/1ntCs6xajR7bT6cgd-fQCuKkoVOIx2oju/view?usp=sharing).

#### Create Data

Go to the CenterPoint's root directory and run

```
# nuScenes
python tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10 --virtual True 
```

if you want to reproduce CenterPoint baseline's results, then also run the following command

```
# nuScenes
python tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10 --virtual False 
```

In the end, the data and info files should be organized as follows

```
# For nuScenes Dataset 
└── CenterPoint
       └── data    
              └── nuScenes 
                     ├── maps          <-- unused
                     |── v1.0-trainval <-- metadata and annotations
                     |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations
                     |── infos_val_10sweeps_withvelo_filter_True.pkl <-- val annotations
                     |── dbinfos_train_10sweeps_withvelo_virtual.pkl <-- GT database info files
                     |── gt_database_10sweeps_withvelo_virtual <-- GT database 
                     |── samples       <-- key frames
                        |── LIDAR_TOP
                        |── LIDAR_TOP_VIRTUAL
                     └── sweeps       <-- frames without annotation
                        |── LIDAR_TOP
                        |── LIDAR_TOP_VIRTUAL
```

#### Train & Evaluate in Command Line

Go to CenterPoint's root directory and use the following command to start a distributed training using 4 GPUs. The models and logs will be saved to ```work_dirs/CONFIG_NAME``` 

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py CONFIG_PATH
```

For distributed testing with 4 gpus,

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth 
```

For testing with one gpu and see the inference time,

```bash
python ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --speed_test 
```
## MODEL ZOO 

We experiment with VoxelNet and PointPillars architectures on nuScenes.

### VoxelNet 
| Model                 | Validation MAP  | Validation NDS  | Link          |
|-----------------------|-----------------|-----------------|---------------|
| [centerpoint_baseline](https://github.com/tianweiy/CenterPoint/blob/master/configs/mvp/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_scale.py) |59.5   | 66.7 | [URL](https://drive.google.com/drive/folders/1abNTNfhbkoPMT-cvNIoDfuveB-2v7oNs?usp=sharing)  |
| [Ours](https://github.com/tianweiy/CenterPoint/blob/master/configs/mvp/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_scale_virtual.py) |**66.0** | **69.9** |  [URL](https://drive.google.com/drive/folders/1HjFv3BZASQk9NscPJku9PBipW610MZ4j?usp=sharing) |  
| [Ours + Two Stage](https://github.com/tianweiy/CenterPoint/blob/master/configs/mvp/nusc_two_stage_base_with_virtual.py) |**67.0** | **70.7** |  [URL](https://drive.google.com/drive/folders/1ecJPj9lzZiwzpVVPDUYHjB6vbUk6dRrO?usp=sharing) |  


### PointPillars
| Model                 | Validation MAP  | Validation NDS  | Link          |
|-----------------------|-----------------|-----------------|---------------|
| [centerpoint_baseline](https://github.com/tianweiy/CenterPoint/blob/master/configs/mvp/nusc_centerpoint_pp_fix_bn_z_scale.py) | 52.4   | 61.5 | [URL](https://drive.google.com/drive/folders/1_Eu1oArVHZ9EgKhNPn1oAJXhR2gE9ZuB?usp=sharing)  |
| [Ours](https://github.com/tianweiy/CenterPoint/blob/master/configs/mvp/nusc_centerpoint_pp_fix_bn_z_scale_virtual.py) |**62.8** | **66.2** |   [URL](https://drive.google.com/drive/folders/1oXz9o8f3mj0VFQl_vSXjoPoSR-cXRRBU?usp=sharing)  |

Test set models and predictions will be updated soon. 

## License

MIT License.
